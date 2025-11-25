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


EPS = 1e-8


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
        # λ_t 参数
        use_dynamic_lambda: bool = True,
        tau1: float = 0.80,  # [2025-11-25] 从0.7提高到0.8，降低lambda对r_t波动的敏感度
        tau2: float = 0.20,  # [2025-11-25] 从0.3降低到0.2，配合lambda_t EMA平滑
        lambda_min: float = 0.1,
        lambda_max: float = 1.0,
        # γ_t 参数
        use_dynamic_gamma: bool = True,
        gamma_initial: float = 0.3,
        gamma_min: float = 0.05,
        gamma_max: float = 0.5,
        tau_n_min: int = 3,
        tau_n_max: int = 25,
    ):
        """
        Args:
            model: BoTorch模型
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
        """
        self.model = model

        # λ_t 配置
        self.use_dynamic_lambda = use_dynamic_lambda
        self.tau1 = tau1
        self.tau2 = tau2
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        # γ_t 配置
        self.use_dynamic_gamma = use_dynamic_gamma
        self.gamma_initial = gamma_initial
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.tau_n_min = int(
            tau_n_min
        )  # Convert to int to handle config parsing floats
        self.tau_n_max = int(
            tau_n_max
        )  # Convert to int to handle config parsing floats

        # 状态缓存
        self._initial_param_vars: Optional[torch.Tensor] = None
        self._current_lambda: float = lambda_max
        self._current_gamma: float = gamma_initial
        self._fitted: bool = False
        self._n_train: int = 0

        # NEW: 参数变化率跟踪（用于新的r_t计算方法）
        self._prev_core_params: Optional[torch.Tensor] = None
        self._initial_param_norm: Optional[float] = None
        self._params_need_update: bool = True  # 标志：参数历史需要更新
        self._cached_r_t: Optional[float] = None  # 缓存的 r_t 值
        self._cached_r_t_n_train: int = -1  # 缓存的 r_t 对应的 n_train
        self._r_t_smoothed: Optional[float] = None  # EMA 平滑后的 r_t 值

        # [2025-11-25] 添加lambda_t EMA平滑，避免大幅回退
        self._prev_lambda: Optional[float] = None  # lambda_t的EMA缓存

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

    def update_training_status(self, n_train: int, fitted: bool) -> None:
        """更新训练状态（由主类调用）

        Args:
            n_train: 当前训练样本数
            fitted: 模型是否已拟合
        """
        import sys
        print(f"\n[update_training_status] 被调用！旧n_train={self._n_train}, 新n_train={n_train}, fitted={fitted}", file=sys.stderr)

        # 如果训练样本数增加，标记参数历史需要更新
        if n_train > self._n_train:
            self._params_need_update = True
            print(f"[update_training_status] 设置 _params_need_update=True", file=sys.stderr)

        self._n_train = n_train
        self._fitted = fitted
        print(f"[update_training_status] 更新后：_n_train={self._n_train}, _fitted={self._fitted}", file=sys.stderr)


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
            import sys
            print(f"[r_t n={self._n_train}] 参数变化率计算失败: {e}, 使用回退方法", file=sys.stderr)
            return self._compute_prediction_variance_fallback(self.model)

    def _compute_param_change_rate(self, model) -> float:
        """主方法：跟踪参数变化率

        计算核心参数（mean_module, covar_module）相对于上次迭代的变化率。

        Returns:
            r_t ∈ [0, 1]
        """
        # 【缓存机制】如果当前训练迭代的 r_t 已计算过，直接返回缓存值
        if self._cached_r_t is not None and self._cached_r_t_n_train == self._n_train:
            return self._cached_r_t

        # 提取核心参数
        current_params = self._extract_core_parameters(model)

        # 首次迭代：最大不确定性
        if self._prev_core_params is None:
            self._prev_core_params = current_params.clone()
            self._initial_param_norm = torch.norm(current_params).item()
            r_t = 1.0
            # 缓存结果
            self._cached_r_t = r_t
            self._cached_r_t_n_train = self._n_train
            import sys
            print(f"[r_t n={self._n_train}] 首次迭代，r_t={r_t:.6f}", file=sys.stderr)
            return r_t

        # 检查参数数量变化（模型结构改变）
        if len(current_params) != len(self._prev_core_params):
            import sys
            print(
                f"[r_t n={self._n_train}] ⚠️ 核心参数数量变化: "
                f"{len(self._prev_core_params)} -> {len(current_params)}, "
                "重置基线",
                file=sys.stderr,
            )
            self._prev_core_params = current_params.clone()
            self._initial_param_norm = torch.norm(current_params).item()
            r_t = 1.0
            self._cached_r_t = r_t
            self._cached_r_t_n_train = self._n_train
            return r_t

        # 计算相对变化
        param_diff = current_params - self._prev_core_params
        current_norm = torch.norm(current_params).item()
        diff_norm = torch.norm(param_diff).item()

        # 避免除零
        norm_denom = max(current_norm, 1e-8)
        change_rate = diff_norm / norm_denom

        # 【关键修复】只有模型重新训练后才更新历史
        # 使用标志判断：_params_need_update 由 update_training_status 设置
        if self._params_need_update:
            self._prev_core_params = current_params.clone()
            self._params_need_update = False  # 清除标志

            # Debug 输出（只在真正更新时打印）
            import sys
            print(
                f"[r_t n={self._n_train}] 参数已更新: 变化率={change_rate:.6f}, "
                f"差异范数={diff_norm:.6e}, 当前范数={current_norm:.6e}",
                file=sys.stderr,
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
            skip_keywords = ['cutpoint', 'outcome_transform', 'standardize', 'variational']
            if any(kw in name.lower() for kw in skip_keywords):
                continue

            # 保留核心参数
            keep_keywords = ['mean_module', 'covar_module']
            if any(kw in name.lower() for kw in keep_keywords):
                params_to_track.append(param.detach().flatten())

        if not params_to_track:
            raise RuntimeError("无法提取核心参数！检查模型是否有 mean_module/covar_module")

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
            if hasattr(actual_model, "train_inputs") and actual_model.train_inputs is not None:
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
            if not hasattr(self, '_initial_pred_var'):
                self._initial_pred_var = avg_var
                import sys
                print(f"[r_t fallback n={self._n_train}] 初始预测方差={avg_var:.6e}", file=sys.stderr)
                return 1.0

            # 更高的方差 → 更高的 r_t
            r_t = avg_var / max(self._initial_pred_var, 1e-8)
            r_t = min(1.0, r_t)

            import sys
            print(
                f"[r_t fallback n={self._n_train}] 平均方差={avg_var:.6e}, "
                f"初始方差={self._initial_pred_var:.6e}, r_t={r_t:.6f}",
                file=sys.stderr,
            )

            return r_t

        except Exception as e:
            import sys
            print(f"[r_t fallback n={self._n_train}] 回退方法失败: {e}, 返回1.0", file=sys.stderr)
            return 1.0

    def compute_lambda(self) -> float:
        """计算动态交互效应权重 λ_t

        分段函数：
        λ_t(r_t) = {
            λ_min,                                          if r_t > τ_1
            λ_min + (λ_max - λ_min)·(τ_1-r_t)/(τ_1-τ_2),  if τ_2 ≤ r_t ≤ τ_1
            λ_max,                                          if r_t < τ_2
        }

        直觉：
        - r_t高（参数不确定，初期）→ 降低交互权重，聚焦主效应（避免过拟合）
        - r_t低（参数已收敛，后期）→ 提高交互权重，挖掘细节（精雕细琢）

        [方案A] 修复r_t计算缺陷：
        - 降低scale系数(2.0→1.0)，解决r_t虚高问题
        - 增强EMA平滑(alpha 0.7→0.85)，减少r_t波动
        - 保持参数驱动的自适应机制

        Returns:
            λ_t ∈ [lambda_min, lambda_max]
        """
        if not self.use_dynamic_lambda:
            return float(self.lambda_max)

        # 使用参数变化率计算r_t（方案A修复已在compute_relative_main_variance中）
        r_t = self.compute_relative_main_variance()

        if r_t > self.tau1:
            lambda_t = self.lambda_min
        elif r_t < self.tau2:
            lambda_t = self.lambda_max
        else:
            # 线性插值
            t_ratio = (self.tau1 - r_t) / (self.tau1 - self.tau2 + EPS)
            lambda_t = self.lambda_min + (self.lambda_max - self.lambda_min) * t_ratio

        # [2025-11-25] 对lambda_t应用EMA平滑，避免大幅回退
        # alpha=0.6: 60%保留历史，40%响应当前值，平衡稳定性和响应性
        if self._prev_lambda is not None:
            lambda_t = 0.6 * self._prev_lambda + 0.4 * lambda_t
        self._prev_lambda = lambda_t  # 更新缓存

        self._current_lambda = float(lambda_t)
        return float(lambda_t)

    def compute_gamma(self) -> float:
        """计算动态覆盖度权重 γ_t

        两级调整：
        1. 基于样本数的线性插值
           - n < τ_n_min: γ = γ_max（样本少，重视覆盖）
           - n ≥ τ_n_max: γ = γ_min（样本多，重视信息）
           - 中间：线性插值

        2. 基于参数方差比的二阶调整
           - r_t > τ_1: γ ↑ 20%（参数不确定，初期，广撒网探索）
           - r_t < τ_2: γ ↓ 20%（参数已收敛，后期，聚焦利用）

        Returns:
            γ_t ∈ [0.05, 1.0]（硬夹值确保稳定性）
        """
        if not self.use_dynamic_gamma or not self._fitted:
            return float(self.gamma_initial)

        try:
            n_train = self._n_train

            # 1. 基于样本数的线性调整
            if n_train < self.tau_n_min:
                gamma_base = self.gamma_max
            elif n_train >= self.tau_n_max:
                gamma_base = self.gamma_min
            else:
                t_ratio = (n_train - self.tau_n_min) / (self.tau_n_max - self.tau_n_min)
                gamma_base = (
                    self.gamma_max - (self.gamma_max - self.gamma_min) * t_ratio
                )

            # 2. 基于r_t的二阶调整
            r_t = self.compute_relative_main_variance()

            if r_t > self.tau1:
                gamma_adjusted = gamma_base * 1.2  # 提高覆盖
            elif r_t < self.tau2:
                gamma_adjusted = gamma_base * 0.8  # 降低覆盖
            else:
                gamma_adjusted = gamma_base

            # 硬夹值确保稳定
            gamma_t = float(np.clip(gamma_adjusted, 0.05, 1.0))
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
