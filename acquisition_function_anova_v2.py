"""
EURAnovaPairAcqf_v2: 改进版 - 混合/融合型局部ANOVA采集函数

改进点（对比v1）:
1. 【问题1修复】分类变量离散采样 - 完全遵守合法类别空间
   - 分类：直接从unique_vals采样，避免高斯污染
   - 整数：高斯扰动后舍入+夹值
   - 连续：保持原有高斯扰动

2. 【问题3增强】序数熵计算数值稳定性
   - 概率规范化：确保Σp_k=1
   - 安全检查：NaN/Inf降级处理
   - 更严格的EPS边界

3. 【问题4增强】动态γ_t权重调整
   - 基于样本数量的渐进式调整
   - 与r_t联动的二阶调整
   - 自适应覆盖/信息平衡

相比v1的优势：
✅ 分类变量处理理论更纯正（100%合法的邻域）
✅ 熵计算更稳定（无NaN/Inf风险）
✅ 采集策略更自适应（动态γ_t）
✅ 向后兼容（API完全相同）
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


class EURAnovaPairAcqf_v2(AcquisitionFunction):
    """改进版EUR ANOVA采集函数 - 混合变量类型优化版本"""

    def __init__(
        self,
        model: Model,
        # 信息/覆盖融合
        gamma: float = 0.3,
        # 主/交互权重
        main_weight: float = 1.0,
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

        self.main_weight = float(main_weight)
        self.pair_weight = float(pair_weight)
        self.local_jitter_frac = float(local_jitter_frac)
        self.local_num = int(local_num)

        # 动态权重参数（λ_t）
        self.use_dynamic_lambda = bool(use_dynamic_lambda)
        self.tau1 = float(tau1)
        self.tau2 = float(tau2)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self._initial_param_vars: Optional[torch.Tensor] = None
        self._current_lambda: float = self.lambda_max

        # 【新增】动态权重参数（γ_t）
        self.use_dynamic_gamma = bool(use_dynamic_gamma)
        self.gamma_max = float(gamma_max)
        self.gamma_min = float(gamma_min)
        self.tau_n_min = int(tau_n_min)
        self.tau_n_max = int(tau_n_max)
        self._current_gamma: float = gamma

        # 交互对解析
        self._pairs: List[Tuple[int, int]] = []
        if interaction_pairs is not None:
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
                    s = str(it).strip()
                    if (s.startswith('"') and s.endswith('"')) or (
                        s.startswith("'") and s.endswith("'")
                    ):
                        s = s[1:-1]
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
        """预计算每个分类维的unique值"""
        if self._X_train_np is None or self.variable_types is None:
            return
        try:
            for dim_idx, vtype in self.variable_types.items():
                if vtype == "categorical" and 0 <= dim_idx < self._X_train_np.shape[1]:
                    self._unique_vals_dict[dim_idx] = np.unique(
                        self._X_train_np[:, dim_idx]
                    )
        except Exception:
            pass

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

        # 按维类型处理
        for k in dims:
            vt = self.variable_types.get(k) if self.variable_types else None

            if vt == "categorical" and k in self._unique_vals_dict:
                # 【关键】分类：离散采样（完全合法）
                unique_vals = self._unique_vals_dict[k]
                if len(unique_vals) > 0:
                    samples = np.random.choice(unique_vals, size=(B, self.local_num))
                    base[:, :, k] = torch.from_numpy(samples).to(dtype=X_can_t.dtype)

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
        """使用Laplace近似提取参数方差（与v1相同）"""
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

            param_vars = []

            for param in params_to_estimate:
                try:
                    param.requires_grad_(True)
                    self.model.train()
                    with torch.enable_grad():
                        posterior = self.model.posterior(X_train)
                        mean = posterior.mean.squeeze(-1)
                        variance = posterior.variance.squeeze(-1)
                        nll = 0.5 * torch.sum(
                            (y_train.squeeze() - mean) ** 2 / (variance + EPS)
                        )

                    grad = torch.autograd.grad(
                        nll,
                        param,
                        create_graph=False,
                        allow_unused=True,
                        retain_graph=True,
                    )[0]

                    if grad is not None:
                        grad_norm = torch.abs(grad.flatten()).mean() + EPS
                        param_var = 1.0 / grad_norm
                        param_vars.append(param_var.expand_as(param).flatten())
                    else:
                        param_vars.append(torch.ones_like(param).flatten())

                except Exception:
                    param_vars.append(torch.ones_like(param).flatten())

            if len(param_vars) == 0:
                return None

            all_param_vars = torch.cat(param_vars).to(device=device, dtype=dtype)
            return all_param_vars

        except Exception:
            return None

    def _compute_dynamic_lambda(self) -> float:
        """计算动态λ_t（与v1相同）"""
        if not self.use_dynamic_lambda:
            return float(self.pair_weight)

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
        """【新增】计算动态γ_t权重"""
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
                raise AssertionError(f"EURAnovaPairAcqf_v2 仅支持 q=1, got q={q}")
            X_flat = X.squeeze(1)
        else:
            B, d = X.shape
            X_flat = X

        X_can_t = self._canonicalize_torch(X_flat)

        # 基线信息 I(x)
        I0 = self._metric(X_can_t)

        # 主效应 Δ_i
        main_contrib = []
        for i in range(d):
            X_i = self._make_local_hybrid(X_can_t, dims=[i])  # 【关键】使用混合扰动
            Ii = self._metric(X_i).view(B, self.local_num).mean(dim=1)
            Di = torch.clamp(Ii - I0, min=0.0)
            main_contrib.append(Di)
        if len(main_contrib) > 0:
            main_sum = torch.stack(main_contrib, dim=1).mean(dim=1)
        else:
            main_sum = torch.zeros_like(I0)

        # 交互 Δ_ij
        pair_contrib = []
        Ei = [None] * d
        for i in range(d):
            if i < len(main_contrib):
                Ei[i] = main_contrib[i] + I0
            else:
                Ei[i] = None
        for i, j in self._pairs:
            X_ij = self._make_local_hybrid(X_can_t, dims=[i, j])  # 【关键】使用混合扰动
            Iij = self._metric(X_ij).view(B, self.local_num).mean(dim=1)
            if Ei[i] is None:
                X_i = self._make_local_hybrid(X_can_t, dims=[i])
                Ei_i = self._metric(X_i).view(B, self.local_num).mean(dim=1)
            else:
                Ei_i = Ei[i]
            if Ei[j] is None:
                X_j = self._make_local_hybrid(X_can_t, dims=[j])
                Ei_j = self._metric(X_j).view(B, self.local_num).mean(dim=1)
            else:
                Ei_j = Ei[j]
            Dij = torch.clamp(Iij - Ei_i - Ei_j + I0, min=0.0)
            pair_contrib.append(Dij)
        if len(pair_contrib) > 0:
            pair_sum = torch.stack(pair_contrib, dim=1).mean(dim=1)
        else:
            pair_sum = torch.zeros_like(I0)

        # 信息项融合（使用动态λ_t）
        lambda_t = self._compute_dynamic_lambda()
        info_raw = self.main_weight * main_sum + lambda_t * pair_sum

        # 覆盖项
        try:
            cov_t = self._compute_coverage_numpy(X_can_t)
        except Exception:
            cov_t = torch.zeros_like(info_raw)

        # 批标准化并融合（使用动态γ_t）
        def _stdz(x: torch.Tensor) -> torch.Tensor:
            mu = x.mean()
            sd = x.std(unbiased=False)
            return (x - mu) / (sd + EPS)

        info_n = _stdz(info_raw)
        cov_n = _stdz(cov_t)

        # 【关键】使用动态γ_t
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
