"""
EURAnovaMultiAcqf: Expected Utility for ANOVA with Multi-Order Interactions

支持任意阶数交互的ANOVA分解采集函数（主效应 + 二阶 + 三阶 + ...）

核心改进：
1. 【模块化架构】所有功能拆分为独立模块，易于维护和测试
2. 【任意阶数支持】通过ANOVA效应引擎支持1-N阶交互
3. 【灵活配置】可独立启用/禁用主效应、二阶、三阶交互
4. 【性能优化】批量评估所有扰动点（20x+加速）

设计公式：
α(x) = α_info(x) + γ_t · COV(x)

其中：
  α_info(x) = w_1·(1/|J|)·∑Δ_j                    # 主效应
            + λ_2·(1/|I_2|)·∑Δ_ij                 # 二阶交互
            + λ_3·(1/|I_3|)·∑Δ_ijk + ...          # 三阶及以上

  λ_t = f(r_t)  # 动态权重（基于参数方差比）
  γ_t = g(n, r_t)  # 覆盖权重（基于样本数）

Example:
    >>> # 只启用主效应
    >>> acqf = EURAnovaMultiAcqf(
    >>>     model,
    >>>     enable_main=True,
    >>>     enable_pairwise=False
    >>> )
    >>>
    >>> # 主效应 + 二阶交互
    >>> acqf = EURAnovaMultiAcqf(
    >>>     model,
    >>>     enable_main=True,
    >>>     interaction_pairs=[(0,1), (2,3)]
    >>> )
    >>>
    >>> # 主效应 + 二阶 + 三阶
    >>> acqf = EURAnovaMultiAcqf(
    >>>     model,
    >>>     enable_main=True,
    >>>     interaction_pairs=[(0,1), (1,2), (2,3)],
    >>>     interaction_triplets=[(0,1,2)],
    >>>     lambda_2=1.0,  # 二阶权重
    >>>     lambda_3=0.5   # 三阶权重
    >>> )
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import warnings
import numpy as np
import torch

from botorch.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform

# 导入模块化组件（支持包导入和直接运行）
try:
    from .modules import (
        ANOVAEffectEngine,
        create_effects_from_config,
        OrdinalMetricsHelper,
        DynamicWeightEngine,
        LocalSampler,
        CoverageHelper,
        parse_interaction_pairs,
        parse_interaction_triplets,
        parse_variable_types,
        validate_interaction_indices,
        DiagnosticsManager,
    )
except ImportError:
    # 直接运行时的绝对导入
    from modules import (
        ANOVAEffectEngine,
        create_effects_from_config,
        OrdinalMetricsHelper,
        DynamicWeightEngine,
        LocalSampler,
        CoverageHelper,
        parse_interaction_pairs,
        parse_interaction_triplets,
        parse_variable_types,
        validate_interaction_indices,
        DiagnosticsManager,
    )

EPS = 1e-8


class EURAnovaMultiAcqf(AcquisitionFunction):
    """Expected Utility Reduction ANOVA Acquisition Function with Multi-Order Interactions

    支持主效应、二阶、三阶及更高阶交互的ANOVA分解采集函数。

    关键特性：
    1. 任意阶数支持（配置驱动）
    2. 模块化架构（易于扩展和测试）
    3. 批量性能优化（一次模型调用）
    4. 动态权重自适应（λ_t、γ_t）
    5. 混合变量类型（分类/整数/连续）
    """

    def __init__(
        self,
        model: Model,
        # ========== 交互阶数配置 ==========
        enable_main: bool = True,  # 是否启用主效应
        interaction_pairs: Optional[Sequence] = None,  # 二阶交互
        interaction_triplets: Optional[Sequence] = None,  # 三阶交互
        enable_pairwise: bool = True,  # 全局开关：是否启用二阶
        enable_threeway: bool = True,  # 全局开关：是否启用三阶
        # ========== 权重参数 ==========
        main_weight: float = 1.0,  # 主效应权重
        lambda_2: Optional[float] = None,  # 二阶权重（None=动态）
        lambda_2_init: Optional[
            float
        ] = None,  # 【新增】λ_2初始值（如果lambda_2=None且use_dynamic_lambda=True时使用）
        lambda_3: Optional[float] = None,  # 三阶权重（None=0.5）
        # 动态λ参数（用于二阶，如果lambda_2=None）
        use_dynamic_lambda: bool = True,
        tau1: float = 0.7,
        tau2: float = 0.3,
        lambda_min: float = 0.1,
        lambda_max: float = 1.0,
        # ========== 覆盖度参数 ==========
        gamma: float = 0.3,
        use_dynamic_gamma: bool = True,
        gamma_max: float = 0.5,
        gamma_min: Optional[float] = None,
        tau_n_min: int = 3,
        tau_n_max: Optional[int] = None,
        total_budget: Optional[int] = None,  # 实验预算（自动配置tau_n_max）
        coverage_method: str = "min_distance",
        # ========== SPS (Skeleton Prediction Stability) 参数 ==========
        use_sps: bool = True,  # 是否使用SPS方法计算r_t
        sps_sensitivity: float = 8.0,  # SPS敏感度系数
        sps_ema_alpha: float = 0.7,  # SPS平滑系数
        tau_safe: float = 0.5,  # Gamma安全刹车阈值
        gamma_penalty_beta: float = 0.3,  # Gamma惩罚强度
        # ========== 融合方式 ==========
        fusion_method: str = "additive",  # 【新增】"additive" 或 "multiplicative"
        # ========== 变量类型 ==========
        variable_types: Optional[Dict[int, str]] = None,
        variable_types_list: Optional[Union[List[str], str]] = None,
        # ========== 局部扰动 ==========
        local_jitter_frac: float = 0.1,
        local_num: int = 4,
        random_seed: Optional[int] = 42,
        # ========== 混合扰动策略参数 ==========
        use_hybrid_perturbation: bool = False,  # 启用混合扰动（≤3水平变量穷举）
        exhaustive_level_threshold: int = 3,  # 穷举阈值
        exhaustive_use_cyclic_fill: bool = True,  # 循环填充
        # ========== 自动计算 local_num ==========
        auto_compute_local_num: bool = False,  # 自动计算local_num（默认False，手动配置）
        auto_local_num_max: int = 12,  # 自动计算上限
        # ========== 调试 ==========
        debug_components: Union[bool, str] = False,
    ) -> None:
        super().__init__(model=model)

        # ========== 解析调试标志 ==========
        if isinstance(debug_components, str):
            self.debug_components = debug_components.strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
        else:
            self.debug_components = bool(debug_components)

        # ========== 解析变量类型 ==========
        if variable_types_list is not None and variable_types is None:
            variable_types = parse_variable_types(variable_types_list)

        self.variable_types = variable_types

        # ========== 解析交互配置 ==========
        self.enable_main = enable_main
        self.enable_pairwise = enable_pairwise
        self.enable_threeway = enable_threeway

        # ========== 融合方式配置 ==========
        if fusion_method not in ("additive", "multiplicative"):
            raise ValueError(
                f"fusion_method must be 'additive' or 'multiplicative', got {fusion_method}"
            )
        self.fusion_method = fusion_method

        self._pairs: List[Tuple[int, int]] = []
        if interaction_pairs is not None and enable_pairwise:
            self._pairs = parse_interaction_pairs(interaction_pairs)

        self._triplets: List[Tuple[int, int, int]] = []
        if interaction_triplets is not None and enable_threeway:
            self._triplets = parse_interaction_triplets(interaction_triplets)

        # ========== 权重参数 ==========
        if main_weight <= 0:
            raise ValueError(f"main_weight must be positive, got {main_weight}")

        self.main_weight = float(main_weight)

        if self.main_weight != 1.0:
            warnings.warn(
                f"main_weight={main_weight} deviates from design formula. "
                "This may be acceptable for specific scenarios."
            )

        # 二阶权重
        self.use_dynamic_lambda_2 = lambda_2 is None
        self.lambda_2 = float(lambda_2) if lambda_2 is not None else lambda_max

        # 【新增】lambda_2_init：用于动态模式下的初始值（默认为lambda_min）
        self.lambda_2_init = (
            float(lambda_2_init) if lambda_2_init is not None else lambda_min
        )

        # 三阶权重（默认0.5，避免过拟合）
        self.lambda_3 = float(lambda_3) if lambda_3 is not None else 0.5

        # ========== 实验预算自适应 ==========
        if tau_n_max is None:
            if total_budget is not None:
                tau_n_max = int(total_budget * 0.7)
                warnings.warn(
                    f"自动配置: total_budget={total_budget} → tau_n_max={tau_n_max}"
                )
            else:
                tau_n_max = 25

        if gamma_min is None:
            if total_budget is not None:
                gamma_min = 0.05 if total_budget < 30 else 0.1
                warnings.warn(
                    f"自动配置: total_budget={total_budget} → gamma_min={gamma_min}"
                )
            else:
                gamma_min = 0.05

        # ========== 初始化模块化组件 ==========

        # 1. 序数模型辅助
        self.ordinal_helper = OrdinalMetricsHelper(model)

        # 2. 动态权重引擎
        # Get bounds from model if available
        bounds = None
        if hasattr(model, 'bounds'):
            bounds = model.bounds
        elif hasattr(model, '_bounds'):
            bounds = model._bounds

        self.weight_engine = DynamicWeightEngine(
            model=model,
            bounds=bounds,
            use_dynamic_lambda=use_dynamic_lambda,
            tau1=tau1,
            tau2=tau2,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            use_dynamic_gamma=use_dynamic_gamma,
            gamma_initial=gamma,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            tau_n_min=tau_n_min,
            tau_n_max=tau_n_max,
            use_sps=use_sps,
            sps_sensitivity=sps_sensitivity,
            sps_ema_alpha=sps_ema_alpha,
            tau_safe=tau_safe,
            gamma_penalty_beta=gamma_penalty_beta,
        )

        # 3. 局部采样器
        self.local_sampler = LocalSampler(
            variable_types=variable_types,
            local_jitter_frac=local_jitter_frac,
            local_num=local_num,
            random_seed=random_seed,
            use_hybrid_perturbation=use_hybrid_perturbation,
            exhaustive_level_threshold=exhaustive_level_threshold,
            exhaustive_use_cyclic_fill=exhaustive_use_cyclic_fill,
            auto_compute_local_num=auto_compute_local_num,
            auto_local_num_max=auto_local_num_max,
        )

        # 4. 覆盖度计算
        self.coverage_helper = CoverageHelper(
            variable_types=variable_types, coverage_method=coverage_method
        )

        # 5. ANOVA效应引擎（延迟初始化，因为需要知道维度数）
        self.anova_engine: Optional[ANOVAEffectEngine] = None

        # 6. 诊断管理器
        self.diagnostics = DiagnosticsManager(debug_components=self.debug_components)

        # ========== 状态缓存 ==========
        self._X_train_np: Optional[np.ndarray] = None
        self._fitted: bool = False
        self._last_hist_n: int = -1
        self._n_dims: Optional[int] = None

        # 保存配置参数（用于诊断）
        self._config = {
            "main_weight": main_weight,
            "lambda_2": self.lambda_2,
            "lambda_2_init": self.lambda_2_init,  # 【新增】
            "lambda_3": self.lambda_3,
            "enable_main": enable_main,
            "enable_pairwise": enable_pairwise,
            "enable_threeway": enable_threeway,
            "fusion_method": self.fusion_method,  # 【新增】
            "n_pairs": len(self._pairs),
            "pairs": self._pairs,
            "n_triplets": len(self._triplets),
            "triplets": self._triplets,
        }

        # 添加动态权重配置
        self._config.update(self.weight_engine.get_diagnostics()["config"])

    # ========== 数据同步 ==========

    def _ensure_fresh_data(self) -> None:
        """同步训练数据并更新所有模块"""
        if not hasattr(self.model, "train_inputs") or self.model.train_inputs is None:
            return

        X_t = self.model.train_inputs[0]
        y_t = getattr(self.model, "train_targets", None)

        if X_t is None or y_t is None:
            return

        n = X_t.shape[0]

        # 首次初始化或数据更新
        if (not self._fitted) or (n != self._last_hist_n):
            # 应用变换（确保与候选点在同一空间）
            X_t_canonical = self._canonicalize_torch(X_t)
            self._X_train_np = X_t_canonical.detach().cpu().numpy()
            self._last_hist_n = n
            self._fitted = True
            self._n_dims = X_t_canonical.shape[1]

            # 更新各模块
            self.local_sampler.update_data(self._X_train_np)
            self.coverage_helper.update_training_data(self._X_train_np)
            self.weight_engine.update_training_status(n, self._fitted)

            # 验证交互索引并过滤越界项
            self._pairs, self._triplets = validate_interaction_indices(
                self._pairs, self._triplets, self._n_dims
            )

            # 更新配置
            self._config["n_pairs"] = len(self._pairs)
            self._config["pairs"] = self._pairs
            self._config["n_triplets"] = len(self._triplets)
            self._config["triplets"] = self._triplets

            # 首次检查序数模型配置
            if not hasattr(self, "_ordinal_check_done"):
                self.ordinal_helper.check_config()
                self._ordinal_check_done = True

    def _canonicalize_torch(self, X: torch.Tensor) -> torch.Tensor:
        """应用模型变换（如果存在）"""
        try:
            tf = getattr(self.model, "transforms", None)
            if tf is None:
                return X
            return tf.transform(X)
        except (AttributeError, RuntimeError) as e:
            # 【修复】已知可接受的异常：记录警告但继续
            warnings.warn(
                f"Failed to apply model transforms: {e}. "
                f"Using untransformed inputs (may affect performance).",
                RuntimeWarning,
            )
            return X
        except Exception as e:
            # 【修复】未知异常：记录详细trace后降级
            warnings.warn(
                f"Unexpected exception in _canonicalize_torch: {type(e).__name__}: {e}. "
                f"This may indicate a bug. Falling back to untransformed inputs.",
                RuntimeWarning,
            )
            return X

    # ========== 信息度量 ==========

    def _metric(self, X_can_t: torch.Tensor) -> torch.Tensor:
        """计算信息度量（序数用熵，回归用方差）"""
        try:
            if self.ordinal_helper.is_ordinal():
                # 序数模型：使用熵
                return self.ordinal_helper.compute_entropy(X_can_t)
            else:
                # 回归模型：使用方差
                with torch.no_grad():
                    posterior = self.model.posterior(X_can_t)
                    var = getattr(posterior, "variance", None)

                    if var is None:
                        return torch.ones(
                            X_can_t.shape[0], dtype=X_can_t.dtype, device=X_can_t.device
                        )

                    # 维度归约
                    while var.dim() > 1 and var.shape[-1] == 1:
                        var = var.squeeze(-1)
                    if var.dim() > 1:
                        var = var.mean(dim=-1)

                    return torch.clamp(var.view(-1), min=EPS)

        except Exception:
            return torch.ones(
                X_can_t.shape[0], dtype=X_can_t.dtype, device=X_can_t.device
            )

    # ========== 核心forward方法 ==========

    @t_batch_mode_transform()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """采集函数评估

        Args:
            X: (B, d) 或 (B, 1, d) 候选点

        Returns:
            (B,) 采集值（越大越好）
        """
        self._ensure_fresh_data()

        if (
            not self._fitted
            or self._X_train_np is None
            or self._X_train_np.shape[0] == 0
        ):
            # 降级：返回随机值
            B = X.shape[0] if X.dim() != 3 else X.shape[0]
            return torch.rand(B, dtype=X.dtype, device=X.device)

        if X.dim() < 3:
            raise ValueError(
                "EURAnovaMultiAcqf expects inputs with shape (..., q, d). "
                "If you are passing (n, d), add a q=1 dimension via X.unsqueeze(1)."
            )

        batch_shape = X.shape[:-2]
        q = X.shape[-2]
        d = X.shape[-1]

        if q != 1:
            raise AssertionError(
                "EURAnovaMultiAcqf currently supports only q=1. "
                "Please pass X with shape (..., 1, d)."
            )

        X_flat = X.reshape(-1, d)

        X_can_t = self._canonicalize_torch(X_flat)

        # ========== 初始化ANOVA引擎（延迟初始化）==========
        if self.anova_engine is None:
            self.anova_engine = ANOVAEffectEngine(
                metric_fn=self._metric, local_sampler=self.local_sampler.sample
            )

        # ========== 构造效应列表 ==========
        effects = create_effects_from_config(
            n_dims=d,
            enable_main=self.enable_main,
            interaction_pairs=self._pairs if self.enable_pairwise else None,
            interaction_triplets=self._triplets if self.enable_threeway else None,
        )

        # ========== 批量计算所有效应 ==========
        results = self.anova_engine.compute_effects(X_can_t, effects)

        # ========== 加权融合 ==========
        # 计算动态权重
        if self.use_dynamic_lambda_2:
            lambda_2_t = self.weight_engine.compute_lambda()
        else:
            lambda_2_t = self.lambda_2

        # 【修复】确保 gamma_t 也被计算并保存
        gamma_t_computed = self.weight_engine.compute_gamma()

        lambda_3_t = self.lambda_3

        # 提取各阶效应
        main_sum = results["aggregated"].get(
            "order_1", torch.zeros_like(results["baseline"])
        )
        pair_sum = results["aggregated"].get(
            "order_2", torch.zeros_like(results["baseline"])
        )
        triplet_sum = results["aggregated"].get(
            "order_3", torch.zeros_like(results["baseline"])
        )

        # 信息项融合
        info_raw = self.main_weight * main_sum

        if self.enable_pairwise and len(self._pairs) > 0:
            info_raw = info_raw + lambda_2_t * pair_sum

        if self.enable_threeway and len(self._triplets) > 0:
            info_raw = info_raw + lambda_3_t * triplet_sum

        # ========== 覆盖项计算 ==========
        try:
            cov_t = self.coverage_helper.compute_coverage(X_can_t)
        except Exception:
            cov_t = torch.zeros_like(info_raw)

        # ========== 批内标准化 ==========
        def _stdz(x: torch.Tensor) -> torch.Tensor:
            mu = x.mean()
            sd = x.std(unbiased=False)
            return (x - mu) / (sd + EPS)

        info_n = _stdz(info_raw)
        cov_n = _stdz(cov_t)

        # ========== 最终融合 (支持加法和乘法) ==========
        gamma_t = self.weight_engine.compute_gamma()

        if self.fusion_method == "multiplicative":
            # 乘法融合: acq = info * (1 + gamma * cov)
            # 优势: 两项平衡，覆盖项不会被信息项淹没
            total = info_n * (1.0 + gamma_t * cov_n)
        else:
            # 加法融合 (默认): acq = info + gamma * cov
            # 传统方法，但覆盖项可能被信息项淹没
            total = info_n + gamma_t * cov_n

        # 平局抖动
        if (total.max() - total.min()) < 1e-9:
            total = total + (1e-3 * torch.rand_like(total))

        # ========== 更新诊断信息 ==========
        self.diagnostics.update_effects(
            main_sum=main_sum,
            pair_sum=pair_sum,
            triplet_sum=triplet_sum,
            info_raw=info_raw,
            cov=cov_t,
        )

        target_shape = batch_shape
        if len(target_shape) == 0:
            return total.view(())
        return total.view(*target_shape)

    # ========== 诊断接口 ==========

    def get_diagnostics(self) -> Dict[str, Any]:
        """获取完整诊断信息"""
        # 【修复】确保数据已同步（这样 _last_hist_n 会被正确更新）
        self._ensure_fresh_data()

        # 【修复】获取 r_t 值
        # 即使模型未训练，也尝试计算 r_t（会返回默认值）
        try:
            r_t = self.weight_engine.compute_relative_main_variance()
        except Exception:
            # 如果计算失败，返回 None
            r_t = None

        diag = self.diagnostics.get_diagnostics(
            lambda_t=self.weight_engine.get_current_lambda(),
            gamma_t=self.weight_engine.get_current_gamma(),
            lambda_2=(
                self.weight_engine.get_current_lambda()
                if self.use_dynamic_lambda_2
                else self.lambda_2
            ),
            lambda_3=self.lambda_3,
            n_train=self._last_hist_n if self._fitted else 0,
            fitted=self._fitted,
            config=self._config,
        )

        # 【修复】添加 r_t 到诊断信息
        diag["r_t"] = r_t

        return diag

    def print_diagnostics(self, verbose: bool = False) -> None:
        """打印诊断信息到控制台"""
        diag = self.get_diagnostics()
        self.diagnostics.print_diagnostics(diag, verbose=verbose)
