"""
EURAnovaPairAcqf: 基于局部 ANOVA/包含-排除近似的分解式信息采集函数（独立实现）。

特性概述：
- 同时支持回归与序数（ordinal）模型：
    - 回归：使用后验方差作为信息度量。
    - 序数：使用类别概率的预测熵作为信息度量（必要时回退到方差）。
- 信息项进行“主效应/指定位交互”的局部分解：
    - 对每个候选点 x，基于在邻域内只扰动第 i 维、只扰动第 j 维、以及同时扰动 (i,j) 两维，
        通过包含-排除公式估计交互信息：Δ_ij ≈ E[I(x_{ij-扰动})] - E[I(x_{i-扰动})] - E[I(x_{j-扰动})] + I(x)。
    - 主效应信息：Δ_i ≈ E[I(x_{i-扰动})] - I(x)。
- 覆盖项：使用 numpy Gower 覆盖（混合型变量适配）。
- 批标准化融合：对信息与覆盖分别做批内标准化后按权重线性合成；并加入微弱抖动避免平局。

配置键（示例 INI [EURAnovaPairAcqf] 区段）：
- interaction_pairs = ["0,1", "0,3"]    # 以特征索引表示的交互对
- main_weight = 0.5                       # 主效应权重
- pair_weight = 1.0                       # 交互项权重
- gamma = 0.3                             # 覆盖项权重
- local_jitter_frac = 0.1                 # 局部扰动尺度 (相对每维训练数据范围)
- local_num = 4                           # 每种扰动的采样数
- variable_types_list = [categorical, categorical, categorical, categorical]
- debug_components = false

注意：
- 该方法为近似的“局部 ANOVA”信息分解，不依赖线性系数，适用于非线性关系；
    但在高维下计算量随维数和交互对数量增加，应按预算调小 local_num。
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


class EURAnovaPairAcqf(AcquisitionFunction):
    def __init__(
        self,
        model: Model,
        # 信息/覆盖融合
        gamma: float = 0.3,
        # 主/交互权重
        main_weight: float = 0.5,
        pair_weight: float = 1.0,
        # 动态权重参数 (λ_t 适应)
        use_dynamic_lambda: bool = True,
        tau1: float = 0.7,
        tau2: float = 0.3,
        lambda_min: float = 0.1,
        lambda_max: float = 1.0,
        # 交互对（以索引字符串或索引二元组给出）
        interaction_pairs: Optional[Sequence[Union[str, Tuple[int, int]]]] = None,
        # 局部扰动参数
        local_jitter_frac: float = 0.1,
        local_num: int = 4,
        # 类型与覆盖设置（复用 V5）
        variable_types: Optional[Dict[int, str]] = None,
        coverage_method: str = "min_distance",
        variable_types_list: Optional[Union[List[str], str]] = None,
        # 调试
        debug_components: Union[bool, str] = False,
    ) -> None:
        super().__init__(model=model)

        self.gamma = float(gamma)
        self.coverage_method = coverage_method

        # 训练数据缓存（覆盖/扰动范围需要）
        self._X_train_np: Optional[np.ndarray] = None
        self._y_train_np: Optional[np.ndarray] = None
        self._fitted: bool = False
        self._last_hist_n: int = -1

        # 变量类型解析（可来自 transforms 或配置）
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

        self.main_weight = float(main_weight)
        self.pair_weight = float(pair_weight)
        self.local_jitter_frac = float(local_jitter_frac)
        self.local_num = int(local_num)

        # 动态权重参数
        self.use_dynamic_lambda = bool(use_dynamic_lambda)
        self.tau1 = float(tau1)  # 相对方差高阈值
        self.tau2 = float(tau2)  # 相对方差低阈值
        self.lambda_min = float(lambda_min)  # 最小交互权重
        self.lambda_max = float(lambda_max)  # 最大交互权重
        self._initial_param_vars: Optional[torch.Tensor] = (
            None  # 初始参数方差（正确做法）
        )
        self._current_lambda: float = self.lambda_max  # 当前动态权重

        self._pairs: List[Tuple[int, int]] = []
        if interaction_pairs is not None:
            # 兼容字符串（单个或多个对），避免被逐字符迭代
            if isinstance(interaction_pairs, str):
                seq: List[Union[str, Tuple[int, int]]] = [interaction_pairs]
            else:
                seq = list(interaction_pairs)
            for it in seq:
                if isinstance(it, (list, tuple)) and len(it) == 2:
                    i, j = int(it[0]), int(it[1])
                    if i != j:
                        self._pairs.append((min(i, j), max(i, j)))
                else:
                    # 支持如 "0,1" 或 "0|1"，容忍外层和内层引号
                    s = str(it).strip()
                    if (s.startswith('"') and s.endswith('"')) or (
                        s.startswith("'") and s.endswith("'")
                    ):
                        s = s[1:-1]
                    # 支持一次性提供多个对，如 "0,1;1,2" 或 "0,1  1,2"
                    pair_strs: List[str]
                    if ";" in s:
                        pair_strs = [p for p in s.split(";") if p.strip()]
                    elif " " in s and "," in s:
                        pair_strs = [p for p in s.split() if p.strip()]
                    else:
                        pair_strs = [s]
                    for ps in pair_strs:
                        if "," in ps:
                            toks = ps.split(",")
                        elif "|" in ps:
                            toks = ps.split("|")
                        else:
                            toks = [ps]
                        if len(toks) >= 2:
                            t0 = toks[0].strip().strip('"').strip("'")
                            t1 = toks[1].strip().strip('"').strip("'")
                            if t0 != "" and t1 != "":
                                i, j = int(t0), int(t1)
                                if i != j:
                                    self._pairs.append((min(i, j), max(i, j)))

        # 分量缓存（调试用）
        self._last_main: Optional[torch.Tensor] = None
        self._last_pair: Optional[torch.Tensor] = None

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
        """同步训练数据，并在模型更新时重置动态权重的基线."""
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

            # 【重要】不重置 _initial_var，让它在第一次模型创建时设置，之后保持不变
            # 这样 r_t 才能真正反映从初始状态到当前状态的学习进度

            self._maybe_infer_variable_types()

    # ---- 基础信息度量：序数用熵，回归用方差 ----
    def _metric(self, X_can_t: torch.Tensor) -> torch.Tensor:
        # 若为序数，使用预测熵；否则使用后验方差
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
                        ent = self._ordinal_entropy_from_mv(mean_r, base_var, cutpoints)
                        return ent
                    # 若无切点，退化为方差
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

                    # reduce
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

    def _make_local(self, X_can_t: torch.Tensor, dims: Sequence[int]) -> torch.Tensor:
        """基于训练数据范围，对指定维做局部高斯扰动，返回 (B*local_num, d) 样本。
        对分类/整数维同样当作连续处理（模型后验在连续域可定义），随后由覆盖项保证合法探索。
        """
        B, d = X_can_t.shape
        rng = self._feature_ranges()
        if rng is None:
            # 若未知范围，以 [0,1] 近似
            mn = torch.zeros(d, dtype=X_can_t.dtype, device=X_can_t.device)
            mx = torch.ones(d, dtype=X_can_t.dtype, device=X_can_t.device)
        else:
            mn = torch.as_tensor(rng[0], dtype=X_can_t.dtype, device=X_can_t.device)
            mx = torch.as_tensor(rng[1], dtype=X_can_t.dtype, device=X_can_t.device)
        span = torch.clamp(mx - mn, min=1e-6)
        sigma = self.local_jitter_frac * span  # (d,)

        # 构造 (B, local_num, d)
        base = X_can_t.unsqueeze(1).repeat(1, self.local_num, 1)
        noise = torch.randn_like(base) * sigma.view(1, 1, -1)
        # 仅在给定维上扰动
        mask = torch.zeros_like(base)
        for k in dims:
            mask[..., k] = 1.0
        X_loc = base + noise * mask
        X_loc = torch.clamp(X_loc, min=mn.view(1, 1, -1), max=mx.view(1, 1, -1))
        return X_loc.reshape(B * self.local_num, d)

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

    def _ordinal_entropy_from_mv(
        self, mean: torch.Tensor, var: torch.Tensor, cutpoints: torch.Tensor
    ) -> torch.Tensor:
        std = torch.sqrt(torch.clamp(var, min=EPS))
        z = (cutpoints.view(1, -1) - mean.view(-1, 1)) / std.view(-1, 1)
        cdfs = self._normal_cdf(z).clamp(EPS, 1 - EPS)
        p0 = cdfs[:, :1]
        p_last = 1.0 - cdfs[:, -1:]
        if cdfs.shape[1] >= 2:
            mids = torch.clamp(cdfs[:, 1:] - cdfs[:, :-1], min=EPS)
            probs = torch.cat([p0, mids, p_last], dim=1)
        else:
            probs = torch.cat([p0, p_last], dim=1)
        probs = torch.clamp(probs, min=EPS, max=1.0)
        entropy = -torch.sum(probs * torch.log(probs), dim=-1)
        return entropy

    # ---- 动态权重计算 (λ_t 适应) ----
    def _compute_relative_main_variance(self) -> float:
        """计算相对主效应参数方差比 r_t - 正确实现.

        设计（来自EUR论文）:
        r_t = (1/|J|) * Σ_j Var_GP[θ_j|D_t] / Var_GP[θ_j|D_0]

        实现策略：
        使用Laplace近似在MAP估计处计算参数的后验方差

        Laplace近似：
        - 在最大后验(MAP)估计θ_MAP处，计算log-likelihood的Hessian
        - 参数的后验方差 ≈ Hessian^-1 的对角线元素
        - 这对于ExactGP足够精确

        返回值 r_t ∈ [0, 1]：
        - r_t ≈ 1.0: 参数不确定（学习初期）
        - r_t ≈ 0.0: 参数确定（学习完成）
        """
        if not self._fitted or self._X_train_np is None:
            return 1.0

        try:
            # 提取当前参数方差
            current_param_vars = self._extract_parameter_variances_laplace()

            if current_param_vars is None or len(current_param_vars) == 0:
                return 1.0

            # 首次调用：记录初始参数方差
            if self._initial_param_vars is None:
                self._initial_param_vars = current_param_vars.clone().detach()
                return 1.0

            # 计算参数方差比 (当前方差 / 初始方差)
            variance_ratios = current_param_vars / (self._initial_param_vars + EPS)

            # 平均比率作为r_t
            r_t = variance_ratios.mean().item()

            # 限制在[0, 1]范围
            r_t = float(max(0.0, min(1.0, r_t)))

            return r_t

        except Exception as e:
            # 如果Laplace近似失败，回退到保守估计
            return 1.0

    def _extract_parameter_variances_laplace(self) -> Optional[torch.Tensor]:
        """使用改进的Laplace近似从GP模型中提取参数方差.

        策略：
        1. 计算模型在当前参数处的log-marginal-likelihood
        2. 从参数的梯度信息估计参数的不确定性
        3. 使用Fisher信息矩阵的对角线作为参数精度的代理

        Fisher信息矩阵 I = E[-d²/dθ² log p(y|X)]
        参数方差 ≈ diag(I^-1)
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

            # 收集所有需要梯度的参数
            params_to_estimate = [p for p in self.model.parameters() if p.requires_grad]

            if len(params_to_estimate) == 0:
                return None

            param_vars = []

            # 对每个参数计算其方差估计
            for param in params_to_estimate:
                try:
                    # 启用梯度
                    param.requires_grad_(True)

                    self.model.train()
                    with torch.enable_grad():
                        # 重新计算后验
                        posterior = self.model.posterior(X_train)
                        mean = posterior.mean.squeeze(-1)
                        variance = posterior.variance.squeeze(-1)

                        # 简化的log-likelihood (忽略常数项)
                        nll = 0.5 * torch.sum(
                            (y_train.squeeze() - mean) ** 2 / (variance + EPS)
                        )

                    # 计算关于此参数的梯度
                    grad = torch.autograd.grad(
                        nll,
                        param,
                        create_graph=False,
                        allow_unused=True,
                        retain_graph=True,
                    )[0]

                    if grad is not None:
                        # 参数的不确定性与梯度的范数成反比
                        # 梯度大 -> 参数确定 -> 方差小
                        # 梯度小 -> 参数不确定 -> 方差大
                        grad_norm = torch.abs(grad.flatten()).mean() + EPS

                        # 方差估计: 1 / (梯度范数)
                        # 乘以标度因子以保证合理的范围
                        param_var = 1.0 / grad_norm

                        # 展平并添加
                        param_vars.append(param_var.expand_as(param).flatten())
                    else:
                        # 如果梯度为None，使用默认值
                        param_vars.append(torch.ones_like(param).flatten())

                except Exception:
                    # 单个参数的计算失败，使用默认值
                    param_vars.append(torch.ones_like(param).flatten())

            if len(param_vars) == 0:
                return None

            # 连接所有参数的方差
            all_param_vars = torch.cat(param_vars).to(device=device, dtype=dtype)

            return all_param_vars

        except Exception as e:
            # 整体失败时返回None
            return None

    def _compute_dynamic_lambda(self) -> float:
        """根据相对主效应方差 r_t 计算动态权重 λ_t.

        设计:
        分段函数：
        ┌─ λ_min              if r_t > τ_1  (主效应未学，优先主效应)
        ├─ λ_min + (λ_max-λ_min)·(τ_1-r_t)/(τ_1-τ_2)  if τ_2 ≤ r_t ≤ τ_1  (过渡)
        └─ λ_max              if r_t < τ_2  (主效应已学，开始探索交互)

        参数含义：
        - τ_1 = 0.7: 主效应学习还不充分的阈值
        - τ_2 = 0.3: 主效应学习充分的阈值
        - λ_min = 0.1: 最小交互权重（主效应阶段）
        - λ_max = 1.0: 最大交互权重（交互阶段）
        """
        if not self.use_dynamic_lambda:
            return float(self.pair_weight)

        r_t = self._compute_relative_main_variance()

        # 分段函数实现
        if r_t > self.tau1:
            # 主效应方差比高，未充分学习 → 抑制交互探索
            lambda_t = self.lambda_min
        elif r_t < self.tau2:
            # 主效应方差比低，已充分学习 → 加强交互探索
            lambda_t = self.lambda_max
        else:
            # 线性插值过渡阶段 [τ_2, τ_1]
            # 当r_t从τ_1→τ_2时，λ从λ_min→λ_max
            t_ratio = (self.tau1 - r_t) / (self.tau1 - self.tau2 + EPS)
            lambda_t = self.lambda_min + (self.lambda_max - self.lambda_min) * t_ratio

        self._current_lambda = float(lambda_t)
        return float(lambda_t)

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
        # 刷新历史数据，以便覆盖项和范围估计
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

        # 规范化（到模型所需域）
        X_can_t = self._canonicalize_torch(X_flat)

        # 基线信息 I(x)
        I0 = self._metric(X_can_t)  # (B,)

        # 主效应 Δ_i：E[I(x_{i-扰动})] - I(x)
        main_contrib = []  # List[(B,)]
        for i in range(d):
            X_i = self._make_local(X_can_t, dims=[i])  # (B*L, d)
            Ii = self._metric(X_i).view(B, self.local_num).mean(dim=1)  # (B,)
            Di = torch.clamp(Ii - I0, min=0.0)
            main_contrib.append(Di)
        if len(main_contrib) > 0:
            main_sum = torch.stack(main_contrib, dim=1).mean(dim=1)  # (B,)
        else:
            main_sum = torch.zeros_like(I0)

        # 交互 Δ_ij：E[I(x_{ij-扰动})] - E[I(x_{i-扰动})] - E[I(x_{j-扰动})] + I(x)
        pair_contrib = []
        # 复用已计算的 E[I(x_{i-扰动})]
        Ei = [None] * d
        for i in range(d):
            if i < len(main_contrib):
                # 反推出 E[I(x_{i-扰动})] = Δ_i + I0
                Ei[i] = main_contrib[i] + I0
            else:
                Ei[i] = None
        for i, j in self._pairs:
            X_ij = self._make_local(X_can_t, dims=[i, j])  # (B*L, d)
            Iij = self._metric(X_ij).view(B, self.local_num).mean(dim=1)  # (B,)
            # 若 Ei/Ej 缺失则现算
            if Ei[i] is None:
                X_i = self._make_local(X_can_t, dims=[i])
                Ei_i = self._metric(X_i).view(B, self.local_num).mean(dim=1)
            else:
                Ei_i = Ei[i]
            if Ei[j] is None:
                X_j = self._make_local(X_can_t, dims=[j])
                Ei_j = self._metric(X_j).view(B, self.local_num).mean(dim=1)
            else:
                Ei_j = Ei[j]
            Dij = torch.clamp(Iij - Ei_i - Ei_j + I0, min=0.0)
            pair_contrib.append(Dij)
        if len(pair_contrib) > 0:
            pair_sum = torch.stack(pair_contrib, dim=1).mean(dim=1)  # (B,)
        else:
            pair_sum = torch.zeros_like(I0)

        # 信息项融合（使用动态λ_t）
        lambda_t = self._compute_dynamic_lambda()
        info_raw = self.main_weight * main_sum + lambda_t * pair_sum  # (B,)

        # 覆盖项（沿用 V5）
        try:
            cov_t = self._compute_coverage_numpy(X_can_t)
        except Exception:
            cov_t = torch.zeros_like(info_raw)

        # 批标准化并融合
        def _stdz(x: torch.Tensor) -> torch.Tensor:
            mu = x.mean()
            # 使用无偏=False 避免 B=1 时返回 NaN 的问题
            sd = x.std(unbiased=False)
            return (x - mu) / (sd + EPS)

        info_n = _stdz(info_raw)
        cov_n = _stdz(cov_t)
        total = info_n + self.gamma * cov_n

        # 平局抖动
        if (total.max() - total.min()) < 1e-9:
            total = total + (1e-3 * torch.rand_like(total))

        if self.debug_components:
            self._last_main = main_sum.detach().cpu()
            self._last_pair = pair_sum.detach().cpu()
            self._last_info = info_raw.detach().cpu()
            self._last_cov = cov_t.detach().cpu()

        return total.view(X_can_t.shape[0])
