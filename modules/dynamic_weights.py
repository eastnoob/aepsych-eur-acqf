"""
动态权重自适应系统

实现两种动态权重：
1. λ_t：交互效应权重（基于参数方差收敛率）
2. γ_t：覆盖度权重（基于样本数量与参数不确定性）

核心思想：
- 实验早期：高λ探索交互，高γ保证覆盖
- 实验后期：低λ聚焦主效应，低γ精细化采样

Example:
    >>> # 初始化
    >>> weight_engine = DynamicWeightEngine(model)
    >>>
    >>> # 计算权重
    >>> lambda_t = weight_engine.compute_lambda()  # 交互效应权重
    >>> gamma_t = weight_engine.compute_gamma()    # 覆盖度权重
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import torch
from botorch.models.model import Model
from loguru import logger


EPS = 1e-8


class SPS_Tracker:
    """Skeleton Prediction Stability (SPS) Tracker

    Measures model convergence by tracking prediction stability on a fixed
    set of "skeleton" points representing the backbone of the design space.

    Key idea: If the GP's predictions on skeleton points (center + extremes)
    are stable, the main effects have converged, and we can explore interactions.

    Attributes:
        skeleton_points: Fixed set of virtual reference points (2d+1 points)
        prev_predictions: Previous iteration's predictions on skeleton points
        sensitivity: Amplification factor for prediction changes (default 8.0)
        ema_alpha: Exponential moving average weight (default 0.7)
        r_t_smoothed: EMA-smoothed stability metric
    """

    def __init__(
        self,
        bounds: torch.Tensor,
        sensitivity: float = 8.0,
        ema_alpha: float = 0.7,
    ):
        """Initialize SPS Tracker

        Args:
            bounds: Tensor of shape (2, d) with [lower_bounds, upper_bounds]
            sensitivity: Scaling factor for tanh (default 8.0)
                Higher values amplify small prediction changes
            ema_alpha: EMA smoothing weight (default 0.7)
                Higher values = more smoothing
        """
        self.bounds = bounds
        self.sensitivity = sensitivity
        self.ema_alpha = ema_alpha

        # Generate skeleton points: center + per-dimension extremes
        self.skeleton_points = self._generate_skeleton_points()

        # State tracking
        self.prev_predictions: Optional[torch.Tensor] = None
        self.r_t_smoothed: Optional[float] = None

    def _generate_skeleton_points(self) -> torch.Tensor:
        """Generate skeleton points: center + 2 extremes per dimension

        Returns:
            Tensor of shape (2*d + 1, d)
        """
        lower = self.bounds[0]  # Shape: (d,)
        upper = self.bounds[1]  # Shape: (d,)
        d = lower.shape[0]

        skeleton_points = []

        # 1. Center point
        center = (lower + upper) / 2.0
        skeleton_points.append(center)

        # 2. Per-dimension extremes
        for dim_idx in range(d):
            # Low extreme: all dims at center, except dim_idx at lower
            low_point = center.clone()
            low_point[dim_idx] = lower[dim_idx]
            skeleton_points.append(low_point)

            # High extreme: all dims at center, except dim_idx at upper
            high_point = center.clone()
            high_point[dim_idx] = upper[dim_idx]
            skeleton_points.append(high_point)

        return torch.stack(skeleton_points)  # Shape: (2*d + 1, d)

    def compute_r_t(self, model: Model) -> float:
        """Compute stability metric r_t based on skeleton prediction changes

        Args:
            model: Trained GP model

        Returns:
            r_t ∈ [0, 1]: 0 = stable (converged), 1 = unstable (not converged)
        """
        try:
            # Get current predictions on skeleton points
            with torch.no_grad():
                posterior = model.posterior(self.skeleton_points)
                current_predictions = posterior.mean.squeeze()  # Shape: (2*d + 1,)

            # First iteration: no previous predictions
            if self.prev_predictions is None:
                self.prev_predictions = current_predictions.clone()
                self.r_t_smoothed = 1.0  # Assume fully unstable initially
                return 1.0

            # Compute relative change
            diff = current_predictions - self.prev_predictions
            delta_t = torch.norm(diff).item() / (
                torch.norm(current_predictions).item() + EPS
            )

            # Apply sensitivity scaling and tanh normalization
            r_t_raw = torch.tanh(torch.tensor(self.sensitivity * delta_t)).item()

            # Apply EMA smoothing
            if self.r_t_smoothed is None:
                r_t = r_t_raw
            else:
                r_t = (
                    self.ema_alpha * self.r_t_smoothed + (1 - self.ema_alpha) * r_t_raw
                )

            # Update state
            self.prev_predictions = current_predictions.clone()
            self.r_t_smoothed = r_t

            return float(r_t)

        except Exception as e:
            logger.error(f"[SPS_Tracker] Error computing r_t: {e}, returning 1.0")
            return 1.0


class DynamicWeightEngine:
    """动态权重计算引擎

    Attributes:
        model: BoTorch模型
        use_dynamic_lambda: 是否启用动态λ_t
        use_dynamic_gamma: 是否启用动态γ_t
        tau1, tau2: r_t阈值（参数方差比）
        lambda_min, lambda_max: λ_t范围
        gamma_min, gamma_max: γ_t范围
        tau_n_min, tau_n_max: 样本数阈值
    """

    def __init__(
        self,
        model: Model,
        bounds: Optional[torch.Tensor] = None,
        # λ_t 参数
        use_dynamic_lambda: bool = True,
        tau1: float = 0.80,
        tau2: float = 0.20,
        lambda_min: float = 0.1,
        lambda_max: float = 1.0,
        # 分段Lambda参数（新增）
        use_piecewise_lambda: bool = False,
        piecewise_phase1_end: int = 35,
        piecewise_phase2_end: int = 50,
        piecewise_lambda_low: float = 0.35,
        piecewise_lambda_high: float = 0.70,
        # γ_t 参数
        use_dynamic_gamma: bool = True,
        gamma_initial: float = 0.3,
        gamma_min: float = 0.05,
        gamma_max: float = 0.5,
        tau_n_min: int = 3,
        tau_n_max: int = 25,
        # SPS (Skeleton Prediction Stability) 参数
        use_sps: bool = True,
        sps_sensitivity: float = 8.0,
        sps_ema_alpha: float = 0.7,
        # Adaptive Gamma Safety Brake 参数
        tau_safe: float = 0.5,
        gamma_penalty_beta: float = 0.3,
    ):
        """
        Args:
            model: BoTorch模型
            bounds: Tensor of shape (2, d) with [lower_bounds, upper_bounds]
            use_dynamic_lambda: 是否启用动态λ_t
            tau1: r_t上阈值（>tau1时降低交互权重）
            tau2: r_t下阈值（<tau2时提高交互权重）
            lambda_min: 最小交互权重（参数已收敛）
            lambda_max: 最大交互权重（参数不确定）
            use_dynamic_gamma: 是否启用动态γ_t
            gamma_initial: 初始覆盖权重
            gamma_min: 最小覆盖权重（样本充足）
            gamma_max: 最大覆盖权重（样本稀少）
            tau_n_min: 样本数下阈值
            tau_n_max: 样本数上阈值
            use_sps: 是否使用SPS方法计算r_t（替代参数变化率）
            sps_sensitivity: SPS敏感度系数（默认8.0）
            sps_ema_alpha: SPS平滑系数（默认0.7）
            tau_safe: Gamma安全刹车阈值（默认0.5）
            gamma_penalty_beta: Gamma惩罚强度（默认0.3）
        """
        self.model = model
        self.bounds = bounds

        # λ_t 配置
        self.use_dynamic_lambda = use_dynamic_lambda
        self.tau1 = tau1
        self.tau2 = tau2
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        # 分段Lambda配置
        self.use_piecewise_lambda = use_piecewise_lambda
        self.piecewise_phase1_end = int(piecewise_phase1_end)
        self.piecewise_phase2_end = int(piecewise_phase2_end)
        self.piecewise_lambda_low = float(piecewise_lambda_low)
        self.piecewise_lambda_high = float(piecewise_lambda_high)

        # γ_t 配置
        self.use_dynamic_gamma = use_dynamic_gamma
        self.gamma_initial = gamma_initial
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.tau_n_min = int(tau_n_min)
        self.tau_n_max = int(tau_n_max)

        # SPS 配置
        self.use_sps = use_sps
        self.tau_safe = tau_safe
        self.gamma_penalty_beta = gamma_penalty_beta

        # 初始化SPS_Tracker（如果启用）
        self.sps_tracker: Optional[SPS_Tracker] = None
        if self.use_sps:
            # # 如果bounds为None，尝试从模型推断
            # if bounds is None:
            #     bounds = self._infer_bounds_from_model(model)

            # if bounds is not None:
            #     self.sps_tracker = SPS_Tracker(
            #         bounds=bounds,
            #         sensitivity=sps_sensitivity,
            #         ema_alpha=sps_ema_alpha,
            #     )
            # else:
            #     import warnings

            #     warnings.warn(
            #         "use_sps=True but bounds=None and could not infer from model, SPS will not be available. "
            #         "Falling back to parameter change rate method.",
            #         UserWarning,
            #     )

            if self.use_sps and bounds is not None:
                self.sps_tracker = SPS_Tracker(
                    bounds=bounds,
                    sensitivity=sps_sensitivity,
                    ema_alpha=sps_ema_alpha,
                )
            elif self.use_sps and bounds is None:
                import warnings

                warnings.warn(
                    "use_sps=True but bounds=None, SPS will not be available. "
                    "Falling back to parameter change rate method.",
                    UserWarning,
                )

        # 状态缓存
        self._initial_param_vars: Optional[torch.Tensor] = None
        self._current_lambda: float = lambda_max
        self._current_gamma: float = gamma_initial
        self._fitted: bool = False
        self._n_train: int = 0

        # 旧方法（参数变化率）的状态缓存（仅当不使用SPS时需要）
        self._prev_core_params: Optional[torch.Tensor] = None
        self._initial_param_norm: Optional[float] = None
        self._params_need_update: bool = True
        self._cached_r_t: Optional[float] = None
        self._cached_r_t_n_train: int = -1
        self._r_t_smoothed: Optional[float] = None

        # 参数验证
        if self.tau1 <= self.tau2:
            raise ValueError(f"tau1 must be > tau2, got tau1={tau1}, tau2={tau2}")
        if self.lambda_max < self.lambda_min:
            raise ValueError(
                f"lambda_max must be >= lambda_min, "
                f"got lambda_max={lambda_max}, lambda_min={lambda_min}"
            )
        if self.gamma_max < self.gamma_min:
            raise ValueError(
                f"gamma_max must be >= gamma_min, "
                f"got gamma_max={gamma_max}, gamma_min={gamma_min}"
            )
        if self.tau_n_max <= self.tau_n_min:
            raise ValueError(
                f"tau_n_max must be > tau_n_min, "
                f"got tau_n_max={tau_n_max}, tau_n_min={tau_n_min}"
            )

    def _infer_bounds_from_model(self, model: Model) -> Optional[torch.Tensor]:
        """尝试从模型推断边界"""
        try:
            # 尝试从模型获取边界
            if hasattr(model, "bounds") and model.bounds is not None:
                return model.bounds
            elif hasattr(model, "_bounds") and model._bounds is not None:
                return model._bounds
            elif hasattr(model, "train_inputs") and model.train_inputs:
                # 从训练数据推断边界
                X_train = model.train_inputs[0]
                if X_train.ndim >= 2:
                    # 计算每维的最小最大值，并添加10%的边距
                    min_vals = X_train.min(dim=0)[0]
                    max_vals = X_train.max(dim=0)[0]
                    range_vals = max_vals - min_vals
                    margin = 0.1 * range_vals
                    lower_bounds = min_vals - margin
                    upper_bounds = max_vals + margin
                    return torch.stack([lower_bounds, upper_bounds])
            else:
                # 默认边界 [0, 1]^d
                if hasattr(model, "train_inputs") and model.train_inputs:
                    dim = model.train_inputs[0].shape[-1]
                else:
                    dim = 6  # 默认6维
                return torch.tensor([[0.0] * dim, [1.0] * dim], dtype=torch.float32)
        except Exception as e:
            import warnings

            warnings.warn(f"Failed to infer bounds from model: {e}")
            return None

    def update_training_status(self, n_train: int, fitted: bool) -> None:
        """更新训练状态（由主类调用）

        Args:
            n_train: 当前训练样本数
            fitted: 模型是否已拟合
        """
        logger.debug(
            f"[WeightEngine] update_training_status called: old_n={self._n_train}, new_n={n_train}, fitted={fitted}"
        )

        # 如果训练样本数增加，标记参数历史需要更新
        if n_train > self._n_train:
            self._params_need_update = True
            logger.debug(
                f"[WeightEngine] n_train increased: {self._n_train} -> {n_train}, _params_need_update=True"
            )

        self._n_train = n_train
        self._fitted = fitted
        logger.debug(
            f"[WeightEngine] Updated state: _n_train={self._n_train}, _fitted={self._fitted}"
        )

    def compute_relative_main_variance(self) -> float:
        """计算相对参数方差比 r_t（使用参数变化率）

        r_t = ||θ_t - θ_{t-1}||₂ / ||θ_t||₂

        其中：
        - θ_t: 当前迭代的核心参数
        - θ_{t-1}: 上一次迭代的核心参数

        直觉：
        - 高变化率 → 参数仍在调整 → 高不确定性 → r_t 高
        - 低变化率 → 参数已收敛 → 低不确定性 → r_t 低

        Returns:
            r_t ∈ [0, 1]
            - r_t ≈ 1.0: 高不确定性（参数快速变化）
            - r_t ≈ 0.0: 低不确定性（参数已收敛）
            - 1.0 if 无法计算（首次迭代或异常情况）
        """
        try:
            return self._compute_param_change_rate(self.model)
        except Exception as e:
            # 回退到预测方差方法
            logger.warning(
                f"[r_t n={self._n_train}] Param change rate failed: {e}, using fallback"
            )
            return self._compute_prediction_variance_fallback(self.model)

    def _compute_param_change_rate(self, model) -> float:
        """主方法：跟踪参数变化率

        计算核心参数（mean_module, covar_module）相对于上次迭代的变化率。

        Returns:
            r_t ∈ [0, 1]
        """
        # 【缓存机制】如果当前训练迭代的 r_t 已计算过，直接返回缓存值
        if self._cached_r_t is not None and self._cached_r_t_n_train == self._n_train:
            logger.debug(f"[r_t n={self._n_train}] Returning cached r_t={self._cached_r_t:.6f}")
            return self._cached_r_t

        # 提取核心参数
        current_params = self._extract_core_parameters(model)

        # 首次迭代：最大不确定性，并保存参数作为基线
        if self._prev_core_params is None:
            self._prev_core_params = current_params.clone()
            self._initial_param_norm = torch.norm(current_params).item()
            r_t = 1.0
            self._r_t_smoothed = r_t  # 初始化EMA状态
            # 缓存结果
            self._cached_r_t = r_t
            self._cached_r_t_n_train = self._n_train
            self._params_need_update = False  # 已更新，清除标志
            logger.debug(f"[r_t n={self._n_train}] First iteration, r_t={r_t:.6f}, saved baseline params")
            return r_t

        # 检查参数数量变化（模型结构改变）
        if len(current_params) != len(self._prev_core_params):
            logger.warning(
                f"[r_t n={self._n_train}] Core param count changed: "
                f"{len(self._prev_core_params)} -> {len(current_params)}, resetting baseline"
            )
            self._prev_core_params = current_params.clone()
            self._initial_param_norm = torch.norm(current_params).item()
            r_t = 1.0
            self._cached_r_t = r_t
            self._cached_r_t_n_train = self._n_train
            return r_t

        # 计算相对变化（当前参数 vs 上次保存的参数）
        param_diff = current_params - self._prev_core_params
        current_norm = torch.norm(current_params).item()
        diff_norm = torch.norm(param_diff).item()

        # 避免除零
        norm_denom = max(current_norm, 1e-8)
        change_rate = diff_norm / norm_denom

        # 【关键修复】每次训练后都更新参数历史
        # 无论 _params_need_update 标志如何，模型参数变化就应该被追踪
        # 这确保 r_t 能正确反映每次模型refit后的参数变化
        self._prev_core_params = current_params.clone()
        self._params_need_update = False  # 清除标志

        # Debug 输出
        logger.debug(
            f"[r_t n={self._n_train}] change_rate={change_rate:.6f}, "
            f"diff_norm={diff_norm:.6e}, current_norm={current_norm:.6e}"
        )

        # [方案A修复1] 去掉×2放大系数,降低r_t虚高
        # 原scale=2.0导致r_t均值0.68,过高→lambda_t过低
        # 改为1.0后预期r_t降至0.30-0.35→lambda_t后期可达0.75-0.80
        r_t_raw = min(1.0, change_rate * 1.0)

        # [方案A修复2] 增强EMA平滑(alpha 0.7→0.85),减少r_t波动
        # 原alpha=0.7导致r_t波动→lambda_t单调性仅65.3%
        # 改为0.85后预期lambda_t单调性提升至85-90%
        if self._r_t_smoothed is None:
            r_t = r_t_raw  # First iteration: no smoothing
        else:
            r_t = 0.85 * self._r_t_smoothed + 0.15 * r_t_raw

        self._r_t_smoothed = r_t  # Update EMA state

        # 缓存结果（关键！）
        self._cached_r_t = r_t
        self._cached_r_t_n_train = self._n_train

        return r_t

    def _extract_core_parameters(self, model) -> torch.Tensor:
        """提取核心参数：mean_module, covar_module

        跳过结构参数：cutpoints, variational_*, outcome_transform

        Returns:
            扁平化的核心参数向量
        """
        # 处理变换模型（如 ParameterTransformedOrdinalGPModel）
        actual_model = model
        if hasattr(model, "_model") and model._model is not None:
            actual_model = model._model
        elif hasattr(model, "model") and model.model is not None:
            actual_model = model.model

        params_to_track = []

        for name, param in actual_model.named_parameters():
            if not param.requires_grad:
                continue

            # 跳过结构参数（关键！）
            skip_keywords = [
                "cutpoint",
                "outcome_transform",
                "standardize",
                "variational",
            ]
            if any(kw in name.lower() for kw in skip_keywords):
                continue

            # 保留核心参数
            keep_keywords = ["mean_module", "covar_module"]
            if any(kw in name.lower() for kw in keep_keywords):
                params_to_track.append(param.detach().flatten())

        if not params_to_track:
            raise RuntimeError(
                "无法提取核心参数！检查模型是否有 mean_module/covar_module"
            )

        return torch.cat(params_to_track)

    def _compute_prediction_variance_fallback(self, model) -> float:
        """回退方法：使用预测方差作为不确定性代理

        在网格上采样，计算平均预测方差。

        Returns:
            r_t ∈ [0, 1]
        """
        try:
            # 处理变换模型
            actual_model = model
            if hasattr(model, "_model") and model._model is not None:
                actual_model = model._model
            elif hasattr(model, "model") and model.model is not None:
                actual_model = model.model

            # 获取输入维度
            if (
                hasattr(actual_model, "train_inputs")
                and actual_model.train_inputs is not None
            ):
                dim = actual_model.train_inputs[0].shape[-1]
            else:
                # 回退：假设6维
                dim = 6

            # 生成网格（20个随机点）
            grid = torch.rand(20, dim)

            # 获取预测方差
            with torch.no_grad():
                posterior = actual_model.posterior(grid)
                var = posterior.variance

            avg_var = var.mean().item()

            # 跟踪初始方差
            if not hasattr(self, "_initial_pred_var"):
                self._initial_pred_var = avg_var
                logger.debug(
                    f"[r_t fallback n={self._n_train}] Initial pred var={avg_var:.6e}"
                )
                return 1.0

            # 更高的方差 → 更高的 r_t
            r_t = avg_var / max(self._initial_pred_var, 1e-8)
            r_t = min(1.0, r_t)

            logger.debug(
                f"[r_t fallback n={self._n_train}] avg_var={avg_var:.6e}, "
                f"init_var={self._initial_pred_var:.6e}, r_t={r_t:.6f}"
            )

            return r_t

        except Exception as e:
            logger.error(
                f"[r_t fallback n={self._n_train}] Fallback failed: {e}, returning 1.0"
            )
            return 1.0

    def compute_lambda(self) -> float:
        """计算动态交互效应权重 λ_t

        支持两种策略：
        1. r_t-based (原策略): 基于模型收敛度动态调整
        2. 分段策略 (新增): 基于样本数分段控制
           - Phase 1 (n < phase1_end): lambda = lambda_low
           - Phase 2 (phase1_end ≤ n < phase2_end): lambda线性增长
           - Phase 3 (n ≥ phase2_end): lambda = lambda_high

        分段函数：
        λ_t(r_t) = {
            λ_min,                                          if r_t > τ_1
            λ_min + (λ_max - λ_min)·(τ_1-r_t)/(τ_1-τ_2),  if τ_2 ≤ r_t ≤ τ_1
            λ_max,                                          if r_t < τ_2
        }

        直觉：
        - r_t高（模型未收敛，初期）→ 降低交互权重，聚焦主效应
        - r_t低（模型已收敛，后期）→ 提高交互权重，挖掘交互

        [SPS Strategy] 使用Skeleton Prediction Stability:
        - 通过骨架点预测稳定性区分"探索"vs"未收敛"
        - 删除lambda_t的EMA平滑（SPS已经平滑r_t，避免双重延迟）
        - 无状态直接映射，lambda_t快速响应r_t变化

        Returns:
            λ_t ∈ [lambda_min, lambda_max]
        """
        if not self.use_dynamic_lambda:
            return float(self.lambda_max)

        # 【新增】分段Lambda策略（基于样本数）
        if self.use_piecewise_lambda:
            n = self._n_train

            if n < self.piecewise_phase1_end:
                # Phase 1: 低lambda，建立主效应
                lambda_t = self.piecewise_lambda_low
            elif n >= self.piecewise_phase2_end:
                # Phase 3: 高lambda，开发交互
                lambda_t = self.piecewise_lambda_high
            else:
                # Phase 2: 线性增长
                phase2_span = self.piecewise_phase2_end - self.piecewise_phase1_end
                progress = (n - self.piecewise_phase1_end) / phase2_span
                lambda_t = (
                    self.piecewise_lambda_low
                    + (self.piecewise_lambda_high - self.piecewise_lambda_low)
                    * progress
                )

            # 边界保护
            lambda_t = np.clip(lambda_t, self.lambda_min, self.lambda_max)
            self._current_lambda = float(lambda_t)
            return float(lambda_t)

        # 【原有】r_t-based策略
        # 计算r_t: 优先使用SPS，回退到参数变化率
        if self.use_sps and self.sps_tracker is not None:
            r_t = self.sps_tracker.compute_r_t(self.model)
        else:
            r_t = self.compute_relative_main_variance()

        # 无状态直接映射（删除EMA，避免双重平滑）
        if r_t > self.tau1:
            lambda_t = self.lambda_min
        elif r_t < self.tau2:
            lambda_t = self.lambda_max
        else:
            # 线性插值
            t_ratio = (self.tau1 - r_t) / (self.tau1 - self.tau2 + EPS)
            lambda_t = self.lambda_min + (self.lambda_max - self.lambda_min) * t_ratio

        self._current_lambda = float(lambda_t)
        return float(lambda_t)

    def compute_gamma(self) -> float:
        """计算动态覆盖度权重 γ_t with Safety Brake

        策略：
        1. 基于样本数的线性衰减（Base Schedule）
           - n < τ_n_min: γ = γ_max（样本少，重视覆盖）
           - n ≥ τ_n_max: γ = γ_min（样本多，重视信息）
           - 中间：线性插值

        2. 安全刹车（Safety Brake）
           如果r_t > tau_safe，说明主效应（全局结构）仍不稳定，
           必须提高gamma强制空间覆盖以稳定模型：

           γ_t = γ_base + β × (r_t - tau_safe)  if r_t > tau_safe

        Returns:
            γ_t ∈ [gamma_min, 1.0]
        """
        if not self.use_dynamic_gamma or not self._fitted:
            return float(self.gamma_initial)

        try:
            n_train = self._n_train

            # 1. 基于样本数的线性衰减
            if n_train < self.tau_n_min:
                gamma_base = self.gamma_max
            elif n_train >= self.tau_n_max:
                gamma_base = self.gamma_min
            else:
                t_ratio = (n_train - self.tau_n_min) / (self.tau_n_max - self.tau_n_min)
                gamma_base = (
                    self.gamma_max - (self.gamma_max - self.gamma_min) * t_ratio
                )

            # 2. 安全刹车：高不确定性时强制覆盖
            # 获取r_t（与lambda_t使用同一来源）
            if self.use_sps and self.sps_tracker is not None:
                r_t = (
                    self.sps_tracker.r_t_smoothed
                    if self.sps_tracker.r_t_smoothed is not None
                    else 1.0
                )
            else:
                r_t = self.compute_relative_main_variance()

            if r_t > self.tau_safe:
                # Panic mode: 增加gamma惩罚
                penalty = self.gamma_penalty_beta * (r_t - self.tau_safe)
                gamma_t = gamma_base + penalty
            else:
                gamma_t = gamma_base

            # 硬夹值确保稳定
            gamma_t = float(np.clip(gamma_t, self.gamma_min, 1.0))
            self._current_gamma = gamma_t
            return gamma_t

        except Exception:
            return float(self.gamma_initial)

    def get_current_lambda(self) -> float:
        """获取最近一次计算的λ_t（用于诊断）"""
        return self._current_lambda

    def get_current_gamma(self) -> float:
        """获取最近一次计算的γ_t（用于诊断）"""
        return self._current_gamma

    def get_diagnostics(self) -> dict:
        """获取动态权重诊断信息

        Returns:
            {
                'lambda_t': 当前λ_t,
                'gamma_t': 当前γ_t,
                'r_t': 参数方差比,
                'n_train': 训练样本数,
                'fitted': 是否已拟合,
                'config': {配置参数}
            }
        """
        r_t = self.compute_relative_main_variance() if self._fitted else None

        return {
            "lambda_t": self._current_lambda,
            "gamma_t": self._current_gamma,
            "r_t": r_t,
            "n_train": self._n_train,
            "fitted": self._fitted,
            "config": {
                "use_dynamic_lambda": self.use_dynamic_lambda,
                "tau1": self.tau1,
                "tau2": self.tau2,
                "lambda_min": self.lambda_min,
                "lambda_max": self.lambda_max,
                "use_dynamic_gamma": self.use_dynamic_gamma,
                "gamma_min": self.gamma_min,
                "gamma_max": self.gamma_max,
                "tau_n_min": self.tau_n_min,
                "tau_n_max": self.tau_n_max,
            },
        }
