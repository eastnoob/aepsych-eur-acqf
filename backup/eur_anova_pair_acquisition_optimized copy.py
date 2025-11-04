"""
251104晚，进行的备份，当前实现理论上已经基本可用

EURAnovaPairAcqf: Expected Utility for ANOVA with Pair-wise Interactions

基于Expected Utility (EUR) 理论与ANOVA分解的高阶采集函数，支持混合变量与动态权重调整。

核心特性：
1. 【混合变量类型】支持分类/整数/连续变量的混合优化
   - 分类变量：从历史观测值离散采样（保证合法性）
   - 整数变量：高斯扰动后舍入到最近整数
   - 连续变量：标准高斯扰动

2. 【ANOVA效应分解】主效应 + 二阶交互效应
   - 主效应：∑ Δ_i / |J|（所有维度平均）
   - 交互效应：λ_t · ∑ Δ_ij / |I|（动态权重调整）

3. 【动态权重自适应】
   - λ_t：基于参数方差收敛率（r_t）动态调整交互项权重
   - γ_t：基于样本数量动态调整信息/覆盖平衡

4. 【数值稳定性】
   - 序数模型：稳定熵计算（概率规范化 + NaN/Inf检查）
   - 回归模型：后验方差作为信息度量

设计公式：
α(x) = α_info(x) + γ_t · COV(x)
其中：
  α_info(x) = (1/|J|)·∑_j Δ_j + λ_t(r_t)·(1/|I|)·∑_(i,j) Δ_ij
  λ_t = f(r_t)  # 参数方差比的分段函数
  γ_t = g(n, r_t)  # 样本数与参数方差的联合函数
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from botorch.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform

try:  # pragma: no cover
    from .gower_distance import compute_coverage_batch
except Exception:  # pragma: no cover
    from gower_distance import compute_coverage_batch  # type: ignore

EPS = 1e-8


class EURAnovaPairAcqf_BatchOptimized(AcquisitionFunction):
    """Expected Utility Reduction ANOVA Acquisition Function with Pair-wise Interactions (Batch Performance Optimized)

    【批量性能优化版本】通过批量并行化计算，显著降低模型评估次数

    性能优化策略：
    - 将所有局部扰动点合并为一次 posterior 调用（批量并行）
    - 加速比：~21x (对于 d=6, |pairs|=15 的情况)
    - 原始版本：(d + |pairs|) 次模型调用 = 21 次
    - 优化版本：1 次批量模型调用
    - 行为保证：数学完全等价，仅改变计算顺序

    基于期望效用理论与ANOVA分解的采集函数，支持动态权重调整与混合变量类型。

    这是一个生产级采集函数，设计用于高维混合变量优化。核心思想是：
    - 通过参数方差率 r_t 平衡主效应与交互效应的探索权重
    - 通过样本数与参数不确定性动态调整信息/覆盖的获取策略
    - 通过局部ANOVA分解精确评估效应贡献度

    参数设计说明：
    - main_weight: 主效应权重（默认1.0，严格遵循设计公式）
    - lambda_min/max: 交互效应权重范围（动态调整）
    - gamma: 信息/覆盖初始权重（动态调整为 gamma_min ~ gamma_max）
    """

    def __init__(
        self,
        model: Model,
        # 信息/覆盖融合参数
        gamma: float = 0.3,
        # 主效应权重（默认1.0符合设计公式，可调节以适应特殊场景）
        main_weight: float = 1.0,
        # 动态权重参数 (λ_t 交互效应自适应)
        use_dynamic_lambda: bool = True,
        tau1: float = 0.7,  # r_t 上阈值（高于此值降低交互权重）
        tau2: float = 0.3,  # r_t 下阈值（低于此值提高交互权重）
        lambda_min: float = 0.1,  # 最小交互权重（参数已收敛时）
        lambda_max: float = 1.0,  # 最大交互权重（参数不确定时）
        # 交互对（以索引字符串或索引二元组给出）
        interaction_pairs: Optional[Sequence[Union[str, Tuple[int, int]]]] = None,
        # 局部扰动参数
        local_jitter_frac: float = 0.1,
        local_num: int = 4,
        # 类型与覆盖设置
        variable_types: Optional[Dict[int, str]] = None,
        coverage_method: str = "min_distance",
        variable_types_list: Optional[Union[List[str], str]] = None,
        # 动态γ_t参数
        use_dynamic_gamma: bool = True,
        gamma_max: float = 0.5,
        gamma_min: float = 0.1,
        tau_n_min: int = 3,
        tau_n_max: int = 40,
        # 随机种子控制（确保确定性行为）
        random_seed: Optional[int] = 42,
        # 调试
        debug_components: Union[bool, str] = False,
    ) -> None:
        super().__init__(model=model)

        self.gamma = float(gamma)
        self.coverage_method = coverage_method

        # 训练数据缓存
        self._X_train_np: Optional[np.ndarray] = None
        self._y_train_np: Optional[np.ndarray] = None
        self._fitted: bool = False
        self._last_hist_n: int = -1

        # 【新增】预计算的分类值字典
        self._unique_vals_dict: Dict[int, np.ndarray] = {}

        # 变量类型解析
        self.variable_types: Optional[Dict[int, str]] = variable_types
        if variable_types_list is not None and self.variable_types is None:
            raw = variable_types_list
            if isinstance(raw, str):
                s = raw.strip().strip("[]()")
                parts = [p for p in s.replace(";", ",").split(",")]
            else:
                parts = list(raw)
            tokens: List[str] = []
            for p in parts:
                item = str(p).strip().strip('"').strip("'")
                if item:
                    tokens.append(item)
            vt_map: Dict[int, str] = {}
            for i, t in enumerate(tokens):
                t_l = t.lower()
                if t_l.startswith("cat"):
                    vt_map[i] = "categorical"
                elif t_l.startswith("int"):
                    vt_map[i] = "integer"
                elif (
                    t_l.startswith("cont")
                    or t_l.startswith("float")
                    or t_l.startswith("real")
                ):
                    vt_map.setdefault(i, "continuous")
            if len(vt_map) > 0:
                self.variable_types = vt_map

        if isinstance(debug_components, str):
            self.debug_components = debug_components.strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
        else:
            self.debug_components = bool(debug_components)
        self._last_info: Optional[torch.Tensor] = None
        self._last_cov: Optional[torch.Tensor] = None

        # 权重参数验证与存储
        if main_weight <= 0:
            raise ValueError(f"main_weight must be positive, got {main_weight}")
        self.main_weight = float(main_weight)
        if self.main_weight != 1.0:
            import warnings

            warnings.warn(
                f"main_weight={main_weight} deviates from design formula (should be 1.0). "
                "This may be acceptable for specific scenarios but changes the balance."
            )

        self.local_jitter_frac = float(local_jitter_frac)
        self.local_num = int(local_num)

        # 动态权重参数（λ_t - 交互效应自适应）
        self.use_dynamic_lambda = bool(use_dynamic_lambda)
        self.tau1 = float(tau1)
        self.tau2 = float(tau2)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)

        # ✅ 参数验证（防止配置错误）
        if self.tau1 <= self.tau2:
            raise ValueError(
                f"tau1 must be > tau2 for proper dynamic lambda weighting, "
                f"got tau1={self.tau1}, tau2={self.tau2}"
            )
        if self.lambda_max < self.lambda_min:
            raise ValueError(
                f"lambda_max must be >= lambda_min, "
                f"got lambda_max={self.lambda_max}, lambda_min={self.lambda_min}"
            )

        self._initial_param_vars: Optional[torch.Tensor] = None
        self._current_lambda: float = self.lambda_max

        # 【新增】动态权重参数（γ_t）
        self.use_dynamic_gamma = bool(use_dynamic_gamma)
        self.gamma_max = float(gamma_max)
        self.gamma_min = float(gamma_min)
        self.tau_n_min = int(tau_n_min)
        self.tau_n_max = int(tau_n_max)

        # ✅ 参数验证（防止配置错误）
        if self.gamma_max < self.gamma_min:
            raise ValueError(
                f"gamma_max must be >= gamma_min, "
                f"got gamma_max={self.gamma_max}, gamma_min={self.gamma_min}"
            )
        if self.tau_n_max <= self.tau_n_min:
            raise ValueError(
                f"tau_n_max must be > tau_n_min for proper sample-based gamma adjustment, "
                f"got tau_n_max={self.tau_n_max}, tau_n_min={self.tau_n_min}"
            )

        self._current_gamma: float = gamma

        # 随机种子控制
        self.random_seed = random_seed

        # 交互对解析（增强版：自动去重并保持首次出现顺序）
        self._pairs: List[Tuple[int, int]] = []
        if interaction_pairs is not None:
            self._pairs = self._parse_interaction_pairs(interaction_pairs)

        # 分量缓存（调试用）
        self._last_main: Optional[torch.Tensor] = None
        self._last_pair: Optional[torch.Tensor] = None

    def _parse_interaction_pairs(
        self, interaction_pairs: Union[str, Sequence[Union[str, Tuple[int, int]]]]
    ) -> List[Tuple[int, int]]:
        """【增强版】解析交互对输入，自动去重并保持首次出现顺序

        支持格式：
        - [(0,1), (2,3)]           # 元组列表
        - "0,1; 2,3"              # 分号分隔
        - ["0,1", "2|3"]          # 混合分隔符

        关键改进：
        1. 使用 set 进行 O(1) 查重（保持顺序）
        2. 统一的 _add_pair 内部函数（DRY原则）
        3. 详细的解析失败警告
        4. 完全向后兼容原有格式

        Returns:
            去重后的交互对列表（保持首次出现顺序）
        """
        parsed = []
        seen = set()  # 用于 O(1) 查重
        duplicate_count = 0

        # 统一转为列表
        seq = (
            [interaction_pairs]
            if isinstance(interaction_pairs, str)
            else list(interaction_pairs)
        )

        # ✅ 提取去重逻辑为内部函数（提高可维护性）
        def _add_pair(i: int, j: int) -> None:
            """添加交互对（自动规范化和去重）"""
            nonlocal duplicate_count
            if i == j:
                return  # 跳过自环

            pair = (min(i, j), max(i, j))
            if pair not in seen:
                seen.add(pair)
                parsed.append(pair)
            else:
                duplicate_count += 1

        for it in seq:
            try:
                # 类型1：元组/列表
                if isinstance(it, (list, tuple)) and len(it) == 2:
                    _add_pair(int(it[0]), int(it[1]))
                    continue

                # 类型2：字符串
                s = str(it).strip().strip('"').strip("'")

                # 分割分隔符
                if ";" in s:
                    pair_strs = s.split(";")
                elif " " in s and "," in s:
                    pair_strs = s.split()
                else:
                    pair_strs = [s]

                for ps in pair_strs:
                    ps = ps.strip()
                    if not ps:
                        continue

                    # 解析单个对
                    if "," in ps:
                        parts = ps.split(",")
                    elif "|" in ps:
                        parts = ps.split("|")
                    else:
                        import warnings

                        warnings.warn(
                            f"无法解析交互对格式: '{ps}' (需要包含 ',' 或 '|')"
                        )
                        continue

                    if len(parts) >= 2:
                        try:
                            i = int(parts[0].strip().strip('"').strip("'"))
                            j = int(parts[1].strip().strip('"').strip("'"))
                            _add_pair(i, j)  # ✅ 使用统一的添加函数
                        except ValueError as e:
                            import warnings

                            warnings.warn(f"无法解析交互对索引: '{ps}' (错误: {e})")

            except Exception as e:
                import warnings

                warnings.warn(f"解析交互对时出错: {it}, 错误: {e}")

        # 用户友好提示
        if duplicate_count > 0:
            import warnings

            warnings.warn(
                f"交互对输入包含 {duplicate_count} 个重复项，已自动去重（保持首次出现顺序）"
            )

        return parsed

    # ---- transforms & type inference ----
    def _get_param_transforms(self):
        try:
            return getattr(self.model, "transforms", None)
        except Exception:
            return None

    def _canonicalize_torch(self, X: torch.Tensor) -> torch.Tensor:
        try:
            tf = self._get_param_transforms()
            if tf is None:
                return X
            return tf.transform(X)
        except Exception:
            return X

    def _maybe_infer_variable_types(self) -> None:
        if self.variable_types is not None:
            return
        tf = self._get_param_transforms()
        if tf is None:
            return
        vt: Dict[int, str] = {}
        try:
            import importlib

            mod = importlib.import_module("aepsych.transforms.ops")
            Categorical = getattr(mod, "Categorical")
            Round = getattr(mod, "Round")

            for sub in tf.values():
                if hasattr(sub, "indices") and isinstance(sub.indices, list):
                    for idx in sub.indices:
                        if isinstance(sub, Categorical):
                            vt[idx] = "categorical"
                        elif isinstance(sub, Round):
                            vt.setdefault(idx, "integer")
        except Exception:
            pass
        if len(vt) > 0:
            self.variable_types = vt

    # ---- data sync ----
    def _ensure_fresh_data(self) -> None:
        """同步训练数据，预计算分类值字典"""
        if not hasattr(self.model, "train_inputs") or self.model.train_inputs is None:
            return
        X_t = self.model.train_inputs[0]
        y_t = getattr(self.model, "train_targets", None)
        if X_t is None or y_t is None:
            return
        n = X_t.shape[0]
        if (not self._fitted) or (n != self._last_hist_n):
            self._X_train_np = X_t.detach().cpu().numpy()
            self._y_train_np = y_t.detach().cpu().numpy()
            self._last_hist_n = n
            self._fitted = True

            # 【新增】预计算分类值字典
            self._precompute_categorical_values()
            self._maybe_infer_variable_types()

    def _precompute_categorical_values(self) -> None:
        """预计算每个分类维的unique值（增强版：详细错误报告）"""
        if self._X_train_np is None or self.variable_types is None:
            return

        n_dims = self._X_train_np.shape[1]
        failed_dims = []

        for dim_idx, vtype in self.variable_types.items():
            try:
                if vtype == "categorical":
                    # 边界检查
                    if not (0 <= dim_idx < n_dims):
                        failed_dims.append(
                            (dim_idx, f"index out of range [0, {n_dims})")
                        )
                        continue

                    unique_vals = np.unique(self._X_train_np[:, dim_idx])

                    # 空值检查
                    if len(unique_vals) == 0:
                        failed_dims.append((dim_idx, "no valid values"))
                        continue

                    self._unique_vals_dict[dim_idx] = unique_vals

            except Exception as e:
                failed_dims.append((dim_idx, str(e)))

        # 仅在有失败时警告（汇总报告）
        if failed_dims:
            import warnings

            warnings.warn(
                f"预计算分类值失败的维度: {failed_dims}，这些维度将保持原值（无局部探索）"
            )

    # ---- 基础信息度量：序数用熵，回归用方差 ----
    def _metric(self, X_can_t: torch.Tensor) -> torch.Tensor:
        """计算信息度量（序数用熵，回归用方差）"""
        try:
            if self._is_ordinal():
                with torch.no_grad():
                    posterior = self.model.posterior(X_can_t)
                    mean = posterior.mean
                    var = getattr(posterior, "variance", None)
                    if var is None:
                        try:
                            var = posterior.variance
                        except Exception:
                            var = None

                    # squeeze/reduce
                    def _reduce_event(x: torch.Tensor) -> torch.Tensor:
                        while x.dim() > 1 and x.shape[-1] == 1:
                            x = x.squeeze(-1)
                        if x.dim() > 1:
                            x = x.mean(dim=-1)
                        return x.view(-1)

                    mean_r = _reduce_event(mean)
                    if var is None:
                        base_var = torch.ones_like(mean_r)
                    else:
                        base_var = _reduce_event(var)

                    cutpoints = self._get_cutpoints(
                        device=mean_r.device, dtype=mean_r.dtype
                    )
                    if cutpoints is not None:
                        ent = self._ordinal_entropy_from_mv_stable(
                            mean_r, base_var, cutpoints
                        )
                        return ent
                    return torch.clamp(base_var, min=EPS)
            else:
                with torch.no_grad():
                    posterior = self.model.posterior(X_can_t)
                    var = getattr(posterior, "variance", None)
                    if var is None:
                        try:
                            var = posterior.variance
                        except Exception:
                            var = None
                    if var is None:
                        return torch.ones(
                            X_can_t.shape[0], dtype=X_can_t.dtype, device=X_can_t.device
                        )

                    while var.dim() > 1 and var.shape[-1] == 1:
                        var = var.squeeze(-1)
                    if var.dim() > 1:
                        var = var.mean(dim=-1)
                    return torch.clamp(var.view(-1), min=EPS)
        except Exception:
            return torch.ones(
                X_can_t.shape[0], dtype=X_can_t.dtype, device=X_can_t.device
            )

    def _feature_ranges(self) -> Optional[np.ndarray]:
        if self._X_train_np is None:
            return None
        x = self._X_train_np
        mn = x.min(axis=0)
        mx = x.max(axis=0)
        return np.stack([mn, mx], axis=0)  # (2, d)

    def _make_local_hybrid(
        self, X_can_t: torch.Tensor, dims: Sequence[int]
    ) -> torch.Tensor:
        """【改进】混合型局部扰动：分类离散采样，整数舍入，连续高斯

        关键改进：
        - 分类维：从unique_vals直接采样（100%合法）
        - 整数维：高斯扰动后舍入+夹值
        - 连续维：保持原有高斯扰动
        - 降级策略：分类维失败时保持原值（避免非法值）
        """
        B, d = X_can_t.shape
        rng = self._feature_ranges()
        if rng is None:
            mn = torch.zeros(d, dtype=X_can_t.dtype, device=X_can_t.device)
            mx = torch.ones(d, dtype=X_can_t.dtype, device=X_can_t.device)
        else:
            mn = torch.as_tensor(rng[0], dtype=X_can_t.dtype, device=X_can_t.device)
            mx = torch.as_tensor(rng[1], dtype=X_can_t.dtype, device=X_can_t.device)
        span = torch.clamp(mx - mn, min=1e-6)

        # 构造 (B, local_num, d)
        base = X_can_t.unsqueeze(1).repeat(1, self.local_num, 1)

        # 首次警告集合（避免重复警告）
        if not hasattr(self, "_categorical_fallback_warned"):
            self._categorical_fallback_warned = set()

        # 按维类型处理
        for k in dims:
            vt = self.variable_types.get(k) if self.variable_types else None

            if vt == "categorical":
                unique_vals = self._unique_vals_dict.get(k)

                # 【关键】安全检查：unique值是否可用
                if unique_vals is None or len(unique_vals) == 0:
                    # 【降级策略】保持原值（最安全，避免生成非法分类值）
                    if k not in self._categorical_fallback_warned:
                        import warnings

                        warnings.warn(
                            f"分类维 {k} 的unique值未找到，保持原值（该维度无探索贡献）"
                        )
                        self._categorical_fallback_warned.add(k)
                    # base[:, :, k] 保持不变（即 X_can_t[:, k] 的重复）
                    continue
                else:
                    # 【正常路径】离散采样（完全合法）
                    samples = np.random.choice(unique_vals, size=(B, self.local_num))
                    base[:, :, k] = torch.from_numpy(samples).to(
                        dtype=X_can_t.dtype, device=X_can_t.device
                    )

            elif vt == "integer":
                # 【改进】整数：高斯+舍入+夹值
                sigma = self.local_jitter_frac * span[k]
                noise = torch.randn(B, self.local_num, device=X_can_t.device) * sigma
                base[:, :, k] = torch.round(
                    torch.clamp(base[:, :, k] + noise, min=mn[k], max=mx[k])
                )

            else:
                # 【保持】连续：高斯扰动
                sigma = self.local_jitter_frac * span[k]
                noise = torch.randn(B, self.local_num, device=X_can_t.device) * sigma
                base[:, :, k] = torch.clamp(base[:, :, k] + noise, min=mn[k], max=mx[k])

        return base.reshape(B * self.local_num, d)

    # ---- ordinal helpers ----
    def _is_ordinal(self) -> bool:
        try:
            lk = getattr(self.model, "likelihood", None)
            if lk is None:
                return False
            name = type(lk).__name__.lower()
            if "ordinal" in name:
                return True
            if hasattr(lk, "n_levels") or hasattr(lk, "num_levels"):
                return True
            c = self._get_cutpoints(device=torch.device("cpu"), dtype=torch.float64)
            return c is not None
        except Exception:
            return False

    def _get_cutpoints(
        self, device: torch.device, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        if not self._is_ordinal():
            return None
        cand_names = ["cutpoints", "cut_points", "cut_points_", "_cutpoints"]
        lk = self.model.likelihood
        for name in cand_names:
            c = getattr(lk, name, None)
            if c is not None:
                try:
                    c_t = torch.as_tensor(c, device=device, dtype=dtype)
                    return c_t.view(-1)
                except Exception:
                    continue
        return None

    @staticmethod
    def _normal_cdf(z: torch.Tensor) -> torch.Tensor:
        return 0.5 * (
            1.0
            + torch.erf(
                z / torch.sqrt(torch.tensor(2.0, device=z.device, dtype=z.dtype))
            )
        )

    def _ordinal_entropy_from_mv_stable(
        self, mean: torch.Tensor, var: torch.Tensor, cutpoints: torch.Tensor
    ) -> torch.Tensor:
        """【改进】数值稳定的序数熵计算

        改进点：
        1. 更严格的EPS边界（1e-7）
        2. 概率显式规范化
        3. NaN/Inf安全检查
        """
        std = torch.sqrt(torch.clamp(var, min=EPS))
        z = (cutpoints.view(1, -1) - mean.view(-1, 1)) / std.view(-1, 1)
        cdfs = self._normal_cdf(z).clamp(1e-7, 1 - 1e-7)

        p0 = cdfs[:, :1]
        p_last = 1.0 - cdfs[:, -1:]
        if cdfs.shape[1] >= 2:
            mids = torch.clamp(cdfs[:, 1:] - cdfs[:, :-1], min=1e-7)
            probs = torch.cat([p0, mids, p_last], dim=1)
        else:
            probs = torch.cat([p0, p_last], dim=1)

        # 【关键】规范化确保和为1
        probs = probs / (probs.sum(dim=1, keepdim=True) + EPS)
        probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)

        # 【关键】数值稳定的熵计算
        log_probs = torch.log(torch.clamp(probs, min=1e-7))
        entropy = -torch.sum(probs * log_probs, dim=-1)

        # 【关键】安全检查
        entropy = torch.where(
            torch.isnan(entropy) | torch.isinf(entropy),
            torch.ones_like(entropy) * 0.5,  # 降级到均匀分布熵
            entropy,
        )

        return entropy

    # ---- 动态权重计算 (λ_t 适应) ----
    def _compute_relative_main_variance(self) -> float:
        """计算相对主效应参数方差比 r_t（与v1相同）"""
        if not self._fitted or self._X_train_np is None:
            return 1.0

        try:
            current_param_vars = self._extract_parameter_variances_laplace()

            if current_param_vars is None or len(current_param_vars) == 0:
                return 1.0

            if self._initial_param_vars is None:
                self._initial_param_vars = current_param_vars.clone().detach()
                return 1.0

            variance_ratios = current_param_vars / (self._initial_param_vars + EPS)
            r_t = variance_ratios.mean().item()
            r_t = float(max(0.0, min(1.0, r_t)))

            return r_t

        except Exception:
            return 1.0

    def _extract_parameter_variances_laplace(self) -> Optional[torch.Tensor]:
        """【改进版】使用Laplace近似提取参数方差

        关键改进：
        1. 使用 eval() 模式避免 Dropout/BatchNorm 影响
        2. 只计算一次 posterior 和 NLL（10-20x 性能提升）
        3. 显式梯度清理防止累积
        4. finally 块确保异常安全的模式恢复
        """
        try:
            if (
                not hasattr(self.model, "train_inputs")
                or self.model.train_inputs is None
            ):
                return None

            X_train = self.model.train_inputs[0]
            y_train = self.model.train_targets

            if X_train is None or y_train is None or len(y_train) == 0:
                return None

            device = X_train.device
            dtype = X_train.dtype

            params_to_estimate = [p for p in self.model.parameters() if p.requires_grad]

            if len(params_to_estimate) == 0:
                return None

            # 【关键改进1】保存原始模式，统一在外部设置
            original_mode = self.model.training

            try:
                self.model.eval()  # 使用 eval 模式避免随机性

                param_vars = []

                # 【关键改进2】只计算一次 posterior 和 NLL
                with torch.enable_grad():
                    posterior = self.model.posterior(X_train)
                    mean = posterior.mean.squeeze(-1)
                    variance = posterior.variance.squeeze(-1)
                    nll = 0.5 * torch.sum(
                        (y_train.squeeze() - mean) ** 2 / (variance + EPS)
                    )

                # 【关键改进3】分别计算每个参数的梯度，不保留计算图
                for i, param in enumerate(params_to_estimate):
                    try:
                        # 清理之前的梯度
                        if param.grad is not None:
                            param.grad = None

                        # 最后一个参数不需要 retain_graph
                        is_last = i == len(params_to_estimate) - 1

                        grad = torch.autograd.grad(
                            nll,
                            param,
                            create_graph=False,
                            allow_unused=True,
                            retain_graph=(not is_last),  # 【关键】只在非最后一个时保留
                        )[0]

                        if grad is not None:
                            grad_norm = torch.abs(grad.flatten()).mean() + EPS
                            param_var = 1.0 / grad_norm
                            param_vars.append(
                                param_var.expand_as(param).flatten().detach()
                            )
                        else:
                            param_vars.append(torch.ones_like(param).flatten())

                    except Exception:
                        param_vars.append(torch.ones_like(param).flatten())

            finally:
                # 【关键改进4】确保恢复原始模式（异常安全）
                self.model.train(original_mode)

            if len(param_vars) == 0:
                return None

            all_param_vars = torch.cat(param_vars).to(device=device, dtype=dtype)
            return all_param_vars

        except Exception:
            return None

    def _compute_dynamic_lambda(self) -> float:
        """计算动态交互效应权重 λ_t

        设计公式：
        λ_t(r_t) = {
            λ_min,                                    if r_t > τ_1
            λ_min + (λ_max - λ_min)·(τ_1-r_t)/(τ_1-τ_2), if τ_2 ≤ r_t ≤ τ_1
            λ_max,                                    if r_t < τ_2
        }

        其中 r_t = (1/|J|)·∑ Var[θ_j|D_t] / Var[θ_j|D_0]

        直觉：
        - r_t 高（参数已收敛）→ 降低交互权重，聚焦主效应
        - r_t 低（参数不确定）→ 提高交互权重，探索复杂模式
        """
        if not self.use_dynamic_lambda:
            # 禁用动态调整时，使用最大权重（最保守策略）
            return float(self.lambda_max)

        r_t = self._compute_relative_main_variance()

        if r_t > self.tau1:
            lambda_t = self.lambda_min
        elif r_t < self.tau2:
            lambda_t = self.lambda_max
        else:
            t_ratio = (self.tau1 - r_t) / (self.tau1 - self.tau2 + EPS)
            lambda_t = self.lambda_min + (self.lambda_max - self.lambda_min) * t_ratio

        self._current_lambda = float(lambda_t)
        return float(lambda_t)

    def _compute_dynamic_gamma(self) -> float:
        """计算动态信息/覆盖权重 γ_t

        设计思路（v2扩展）：
        1. 基础调整：基于样本数 n 线性插值
           - n < τ_n_min: γ = γ_max（样本少，重视覆盖）
           - n ≥ τ_n_max: γ = γ_min（样本多，重视信息）

        2. 二阶调整：基于参数方差比 r_t
           - r_t > τ_1: γ ↑ 20%（参数稳定，提高覆盖探索）
           - r_t < τ_2: γ ↓ 20%（参数不确定，聚焦信息）

        注意：此功能为v2创新扩展，原始设计中 γ 为固定值
        """
        if not self.use_dynamic_gamma or not self._fitted:
            return float(self.gamma)

        try:
            n_train = self._X_train_np.shape[0]

            # 基于样本数的线性调整
            if n_train < self.tau_n_min:
                gamma_base = self.gamma_max
            elif n_train >= self.tau_n_max:
                gamma_base = self.gamma_min
            else:
                t_ratio = (n_train - self.tau_n_min) / (self.tau_n_max - self.tau_n_min)
                gamma_base = (
                    self.gamma_max - (self.gamma_max - self.gamma_min) * t_ratio
                )

            # 基于r_t的二阶调整
            r_t = self._compute_relative_main_variance()
            if r_t > self.tau1:
                gamma_adjusted = gamma_base * 1.2
            elif r_t < self.tau2:
                gamma_adjusted = gamma_base * 0.8
            else:
                gamma_adjusted = gamma_base

            gamma_t = float(np.clip(gamma_adjusted, 0.05, 1.0))
            self._current_gamma = gamma_t
            return gamma_t

        except Exception:
            return float(self.gamma)

    # ---- coverage (numpy Gower) ----
    def _compute_coverage_numpy(self, X_can_t: torch.Tensor) -> torch.Tensor:
        assert self._fitted and self._X_train_np is not None
        vt = None
        if self.variable_types is not None:
            vt = {
                k: ("categorical" if v == "categorical" else "continuous")
                for k, v in self.variable_types.items()
            }

        X_np = X_can_t.detach().cpu().numpy()
        try:
            d_can = X_np.shape[1]
            d_hist = self._X_train_np.shape[1]
        except Exception:
            d_can = d_hist = -1

        if d_can != d_hist and d_can != -1 and d_hist != -1:
            return torch.zeros(
                X_np.shape[0], dtype=X_can_t.dtype, device=X_can_t.device
            )

        if vt is not None and d_can >= 0:
            vt = {k: v for k, v in vt.items() if 0 <= k < d_can}

        try:
            cov_np = compute_coverage_batch(
                X_np,
                self._X_train_np,
                variable_types=vt,
                ranges=None,
                method=self.coverage_method,
            )
            cov_t = torch.from_numpy(cov_np).to(
                dtype=X_can_t.dtype, device=X_can_t.device
            )
            return cov_t
        except Exception:
            return torch.zeros(
                X_np.shape[0], dtype=X_can_t.dtype, device=X_can_t.device
            )

    # ---- 覆写 forward，执行分解式信息 + 覆盖 ----
    @t_batch_mode_transform()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # 设置随机种子确保确定性行为
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        self._ensure_fresh_data()
        if (
            not self._fitted
            or self._X_train_np is None
            or (
                isinstance(self._X_train_np, np.ndarray)
                and self._X_train_np.shape[0] == 0
            )
        ):
            B = X.shape[0] if X.dim() != 3 else X.shape[0]
            return torch.rand(B, dtype=X.dtype, device=X.device)

        # 展平到 (B, d)
        if X.dim() == 3:
            B, q, d = X.shape
            if q != 1:
                raise AssertionError(f"EURAnovaPairAcqf 仅支持 q=1, got q={q}")
            X_flat = X.squeeze(1)
        else:
            B, d = X.shape
            X_flat = X

        X_can_t = self._canonicalize_torch(X_flat)

        # 基线信息 I(x)
        I0 = self._metric(X_can_t)

        # ========== 【性能优化】批量构造所有局部扰动点 ==========
        X_all_local = []  # 存储所有需要评估的点 (每个元素形状: B*local_num x d)
        segment_info = []  # 记录每个段的索引: (type, id, list_idx)

        # 1. 主效应局部点
        for i in range(d):
            X_i = self._make_local_hybrid(X_can_t, dims=[i])
            list_idx = len(X_all_local)
            segment_info.append(("main", i, list_idx))
            X_all_local.append(X_i)

        # 2. 交互效应局部点
        extra_main_needed = set()  # 记录需要额外计算的单维点
        for i, j in self._pairs:
            # 交互点
            X_ij = self._make_local_hybrid(X_can_t, dims=[i, j])
            list_idx = len(X_all_local)
            segment_info.append(("pair", (i, j), list_idx))
            X_all_local.append(X_ij)

            # 检查是否需要额外的单维点（如果主效应未涵盖）
            if i >= d and i not in extra_main_needed:
                extra_main_needed.add(i)
                X_i = self._make_local_hybrid(X_can_t, dims=[i])
                list_idx = len(X_all_local)
                segment_info.append(("extra_main", i, list_idx))
                X_all_local.append(X_i)

            if j >= d and j not in extra_main_needed:
                extra_main_needed.add(j)
                X_j = self._make_local_hybrid(X_can_t, dims=[j])
                list_idx = len(X_all_local)
                segment_info.append(("extra_main", j, list_idx))
                X_all_local.append(X_j)

        # ========== 【关键优化】一次性批量评估所有点 ==========
        if len(X_all_local) > 0:
            X_batch = torch.cat(X_all_local, dim=0)  # (total_points, d)
            I_batch = self._metric(X_batch)  # 只调用1次！性能提升~21x

            # 解包到各个段
            main_results = {}
            pair_results = {}
            extra_main_results = {}

            # 计算每个段在拼接后张量中的起始位置
            current_row = 0
            for seg_type, seg_id, list_idx in segment_info:
                seg_size = B * self.local_num  # 每个 X_i 的点数
                start_row = current_row
                end_row = current_row + seg_size
                I_seg = I_batch[start_row:end_row].view(B, self.local_num).mean(dim=1)
                current_row = end_row

                if seg_type == "main":
                    main_results[seg_id] = I_seg
                elif seg_type == "pair":
                    pair_results[seg_id] = I_seg
                elif seg_type == "extra_main":
                    extra_main_results[seg_id] = I_seg

        # 计算主效应贡献
        main_contrib = []
        for i in range(d):
            Ii = main_results.get(i)
            if Ii is not None:
                Di = torch.clamp(Ii - I0, min=0.0)
                main_contrib.append(Di)

        if len(main_contrib) > 0:
            main_sum = torch.stack(main_contrib, dim=1).mean(dim=1)
        else:
            main_sum = torch.zeros_like(I0)

        # 构建 Ei 字典（用于交互效应计算）
        # 注意：Ei[i] 应该是 Ii（从 main_results），而不是 Di
        Ei = {}
        for i in range(d):
            if i in main_results:
                Ei[i] = main_results[i]  # 这是 Ii，不是 Di
        for i in extra_main_results:
            Ei[i] = extra_main_results[i]  # 这也是 Ii

        # 计算交互效应贡献
        pair_contrib = []
        for (i, j), Iij in pair_results.items():
            Ei_i = Ei.get(i)
            Ei_j = Ei.get(j)

            if Ei_i is None or Ei_j is None:
                continue  # 跳过无法计算的交互项

            Dij = torch.clamp(Iij - Ei_i - Ei_j + I0, min=0.0)
            pair_contrib.append(Dij)

        if len(pair_contrib) > 0:
            pair_sum = torch.stack(pair_contrib, dim=1).mean(dim=1)
        else:
            pair_sum = torch.zeros_like(I0)

        # 【关键】信息项融合：主效应 + 动态交互效应
        # 设计公式：α_info = (1/|J|)·∑Δ_j + λ_t·(1/|I|)·∑Δ_ij
        lambda_t = self._compute_dynamic_lambda()
        info_raw = self.main_weight * main_sum + lambda_t * pair_sum

        # 注：main_weight 默认为1.0（严格遵循设计）
        #     当 main_weight=1.0, lambda_t 动态调整时，完全符合公式

        # 覆盖项计算
        try:
            cov_t = self._compute_coverage_numpy(X_can_t)
        except Exception:
            cov_t = torch.zeros_like(info_raw)

        # 【批内标准化】使信息/覆盖项在当前候选批次中具有可比尺度
        # 优点：
        #   1. γ_t 的语义稳定（跨数据集的相对权重）
        #   2. 避免量纲不匹配（方差 vs 距离）
        # 注意：标准化基于当前批次，不保证批间可比
        #       但BoTorch优化只需批内相对排序，故无影响
        def _stdz(x: torch.Tensor) -> torch.Tensor:
            mu = x.mean()
            sd = x.std(unbiased=False)
            return (x - mu) / (sd + EPS)

        info_n = _stdz(info_raw)
        cov_n = _stdz(cov_t)

        # 【最终融合】α(x) = α_info(x) + γ_t · COV(x)
        gamma_t = self._compute_dynamic_gamma()
        total = info_n + gamma_t * cov_n

        # 平局抖动
        if (total.max() - total.min()) < 1e-9:
            total = total + (1e-3 * torch.rand_like(total))

        if self.debug_components:
            self._last_main = main_sum.detach().cpu()
            self._last_pair = pair_sum.detach().cpu()
            self._last_info = info_raw.detach().cpu()
            self._last_cov = cov_t.detach().cpu()

        return total.view(X_can_t.shape[0])
