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
        tau1: float = 0.7,
        tau2: float = 0.3,
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
        self._n_train = n_train
        self._fitted = fitted
        print(f"[update_training_status] 更新后：_n_train={self._n_train}, _fitted={self._fitted}", file=sys.stderr)

    def extract_parameter_variances_laplace(self) -> Optional[torch.Tensor]:
        """使用Laplace近似提取参数方差

        Laplace近似：Var[θ] ≈ 1 / |∇²_θ log p(θ|D)|

        ⚠️ 简化版本：用梯度范数的倒数近似Hessian的对角元素
        Var[θ_i] ≈ 1 / |∇_θi NLL|

        【重要限制】：
        1. **简化近似**：只使用一阶梯度而非完整Hessian矩阵
           - 完整版：需计算O(P²)的二阶导数矩阵
           - 简化版：仅计算O(P)的一阶梯度范数倒数
           - 代价：可能低估参数间的协方差影响
           - ✅ **已修复**：现在使用完整 Gaussian NLL（包含 log(σ²) 项）

        2. **CPU执行**：当前实现在CPU上运行（autograd.grad不自动继承设备）
           - GPU模型的梯度计算仍在GPU，但会自动转移到CPU
           - Multi-GPU场景：只使用DataParallel主设备
           - 性能：对于大模型（>1M参数），可能增加~100ms延迟

        3. **内存占用**：循环计算每个参数的梯度
           - 使用retain_graph=True保持计算图直到最后一个参数
           - 峰值显存 ≈ 2× 单次forward显存
           - 对于极大模型，考虑分批计算或禁用dynamic_lambda

        【替代方案】：
        - 如需完整Hessian：考虑使用Laplace库（laplace-torch）
        - 如内存受限：设置use_dynamic_lambda=False使用固定λ
        - 如速度优先：减小模型参数量或降低更新频率

        性能优化：
        1. 使用eval()模式避免Dropout/BN影响
        2. 只计算一次posterior和NLL
        3. 显式梯度清理防止累积
        4. finally块确保异常安全的模式恢复

        Returns:
            (P,) 张量，P为参数总数
            None if 提取失败（回退到r_t=1.0）
        """
        try:
            # 【修复】处理变换模型（如 ParameterTransformedOrdinalGPModel）
            # 优先使用内部模型（_model）的训练数据
            actual_model = self.model
            if hasattr(self.model, "_model") and self.model._model is not None:
                actual_model = self.model._model
            elif hasattr(self.model, "model") and self.model.model is not None:
                # 有些变换模型使用 .model 而不是 ._model
                actual_model = self.model.model

            # 【调试】第一次执行时打印模型信息
            if not hasattr(self, "_debug_printed"):
                import sys
                print(f"[DEBUG dynamic_weights] 模型类型: {type(self.model).__name__}", file=sys.stderr)
                print(f"[DEBUG dynamic_weights] actual_model 类型: {type(actual_model).__name__}", file=sys.stderr)
                print(f"[DEBUG dynamic_weights] has train_inputs: {hasattr(actual_model, 'train_inputs')}", file=sys.stderr)
                if hasattr(actual_model, 'train_inputs'):
                    print(f"[DEBUG dynamic_weights] train_inputs is None: {actual_model.train_inputs is None}", file=sys.stderr)
                self._debug_printed = True

            if (
                not hasattr(actual_model, "train_inputs")
                or actual_model.train_inputs is None
            ):
                return None

            X_train = actual_model.train_inputs[0]
            y_train = actual_model.train_targets

            if X_train is None or y_train is None or len(y_train) == 0:
                if not hasattr(self, "_debug_printed_fail1"):
                    import sys
                    print(f"[DEBUG] 失败点1: X_train is None: {X_train is None}, y_train is None: {y_train is None}", file=sys.stderr)
                    if y_train is not None:
                        print(f"[DEBUG] 失败点1: len(y_train)={len(y_train)}", file=sys.stderr)
                    self._debug_printed_fail1 = True
                return None

            device = X_train.device
            dtype = X_train.dtype

            # 【修复】从实际模型获取参数，过滤掉结构参数（如cutpoints）
            # 使用 named_parameters 以便识别参数类型
            params_to_estimate = []
            param_names = []

            for name, param in actual_model.named_parameters():
                if not param.requires_grad:
                    continue

                # 【智能过滤】只保留核心建模参数（反映效应的理解）
                # 跳过结构参数：
                # - cutpoints: Ordinal模型的阈值参数
                # - variational_*: Variational GP的近似参数（随数据增长）
                # - outcome_transform: 数据归一化参数
                skip_keywords = ['cutpoint', 'outcome_transform', 'standardize', 'variational']
                if any(keyword in name.lower() for keyword in skip_keywords):
                    continue

                # 只保留核心参数：
                # - mean_module: 均值参数（反映基线）
                # - covar_module: 协方差参数（lengthscale等，反映效应影响范围）
                # - likelihood (除cutpoints外): 噪声参数
                keep_keywords = ['mean_module', 'covar_module', 'likelihood']
                if any(keyword in name.lower() for keyword in keep_keywords):
                    params_to_estimate.append(param)
                    param_names.append(name)

            # Debug: 打印过滤后的参数列表（每次都打印，用于诊断）
            import sys
            if self._n_train <= 2:  # 只在前几次迭代打印
                all_param_names = [name for name, _ in actual_model.named_parameters()]
                print(f"[DEBUG n={self._n_train}] 所有参数数: {len(all_param_names)}", file=sys.stderr)
                print(f"[DEBUG n={self._n_train}] 核心参数数: {len(params_to_estimate)}", file=sys.stderr)
                print(f"[DEBUG n={self._n_train}] 所有参数名: {all_param_names}", file=sys.stderr)
                print(f"[DEBUG n={self._n_train}] 核心参数名: {param_names}", file=sys.stderr)

            if len(params_to_estimate) == 0:
                if not hasattr(self, "_debug_printed_fail2"):
                    import sys
                    print(f"[DEBUG] 失败点2: 无可训练参数", file=sys.stderr)
                    self._debug_printed_fail2 = True
                return None

            if not hasattr(self, "_debug_printed_success"):
                import sys
                print(f"[DEBUG] 成功: X_train.shape={X_train.shape}, params={len(params_to_estimate)}", file=sys.stderr)
                self._debug_printed_success = True

            # 保存原始模式
            original_mode = actual_model.training

            try:
                actual_model.eval()  # 使用eval模式避免随机性

                param_vars = []

                # 只计算一次posterior和NLL
                with torch.enable_grad():
                    try:
                        posterior = actual_model.posterior(X_train)
                        mean = posterior.mean.squeeze(-1)
                        variance = posterior.variance.squeeze(-1)

                        # 【修复】完整的 Gaussian NLL 包含 log(variance) 项
                        # NLL = 0.5 * Σ[log(σ²) + (y-μ)²/σ²]
                        # 之前缺少 log 项会减弱梯度对方差的响应
                        nll = 0.5 * torch.sum(
                            torch.log(variance + EPS)
                            + (y_train.squeeze() - mean) ** 2 / (variance + EPS)
                        )
                    except Exception as posterior_err:
                        if not hasattr(self, "_debug_printed_fail3"):
                            import sys
                            print(f"[DEBUG] 失败点3: posterior计算失败: {posterior_err}", file=sys.stderr)
                            self._debug_printed_fail3 = True
                        return None

                # 分别计算每个参数的梯度
                for i, param in enumerate(params_to_estimate):
                    try:
                        # 清理之前的梯度
                        if param.grad is not None:
                            param.grad = None

                        # 最后一个参数不需要retain_graph
                        is_last = i == len(params_to_estimate) - 1

                        grad = torch.autograd.grad(
                            nll,
                            param,
                            create_graph=False,
                            allow_unused=True,
                            retain_graph=(not is_last),
                        )[0]

                        if grad is not None:
                            grad_flat = grad.flatten()

                            # 【关键修改】保存梯度RMS本身，而不是方差
                            # 这样可以在不同时刻计算相对方差时使用原始梯度信息
                            # RMS梯度 = sqrt(mean(grad^2))
                            grad_rms = torch.sqrt((grad_flat ** 2).mean() + EPS)

                            # Debug: 打印前几个参数的梯度
                            if i < 3 and not hasattr(self, f"_debug_printed_grad_{i}"):
                                import sys
                                print(f"[DEBUG 梯度 参数{i}] 形状: {param.shape}, RMS梯度: {grad_rms.item():.6e}", file=sys.stderr)
                                setattr(self, f"_debug_printed_grad_{i}", True)

                            # 存储梯度平方（精度的代理），后续会转换为方差
                            param_vars.append(
                                (grad_rms ** 2).expand_as(param).flatten().detach()
                            )
                        else:
                            param_vars.append(torch.ones_like(param).flatten())

                    except Exception as grad_err:
                        if i == 0 and not hasattr(self, "_debug_printed_fail4"):
                            import sys
                            print(f"[DEBUG] 失败点4: 梯度计算失败 (参数{i}): {grad_err}", file=sys.stderr)
                            self._debug_printed_fail4 = True
                        param_vars.append(torch.ones_like(param).flatten())

            finally:
                # 确保恢复原始模式（异常安全）
                if original_mode:
                    actual_model.train()
                else:
                    actual_model.eval()

            if len(param_vars) == 0:
                return None

            all_param_precisions = torch.cat(param_vars).to(device=device, dtype=dtype)

            # Debug: 每次都打印参数精度统计（移除hasattr限制）
            import sys
            print(f"[精度 n={self._n_train}] 总参数: {all_param_precisions.shape[0]}, 均值={all_param_precisions.mean().item():.6e}, std={all_param_precisions.std().item():.6e}", file=sys.stderr)
            print(f"[精度 n={self._n_train}] 范围: [{all_param_precisions.min().item():.6e}, {all_param_precisions.max().item():.6e}]", file=sys.stderr)

            # 【诊断增强】打印前10个和后10个精度值，用于对比不同迭代
            if all_param_precisions.shape[0] >= 20:
                print(f"[精度 n={self._n_train}] 前10个值: {all_param_precisions[:10].tolist()}", file=sys.stderr)
                print(f"[精度 n={self._n_train}] 后10个值: {all_param_precisions[-10:].tolist()}", file=sys.stderr)
            else:
                print(f"[精度 n={self._n_train}] 所有值: {all_param_precisions.tolist()}", file=sys.stderr)

            return all_param_precisions

        except Exception as e:
            if not hasattr(self, "_debug_printed_exception"):
                import sys
                import traceback
                print(f"[DEBUG] 外层异常捕获: {type(e).__name__}: {e}", file=sys.stderr)
                print(f"[DEBUG] Traceback:", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                self._debug_printed_exception = True
            return None

    def compute_relative_main_variance(self) -> float:
        """计算相对参数方差比 r_t

        r_t = mean(Var[θ_j | D_t] / Var[θ_j | D_0])

        其中：
        - D_0: 初始数据（首次计算时）
        - D_t: 当前数据

        Returns:
            r_t ∈ [0, 1]
            1.0 if 无法计算（安全降级）
        """
        import sys
        print(f"\n[compute_relative_main_variance] 被调用！_n_train={self._n_train}, _fitted={self._fitted}, _initial_param_vars={'None' if self._initial_param_vars is None else 'exists'}", file=sys.stderr)

        if not self._fitted or self._n_train == 0:
            return 1.0

        try:
            # 现在extract_parameter_variances_laplace返回的是precisions (grad^2)
            current_precisions = self.extract_parameter_variances_laplace()

            import sys
            print(f"[r_t n={self._n_train}] extract返回后: current_precisions is None={current_precisions is None}, len={len(current_precisions) if current_precisions is not None else 'N/A'}", file=sys.stderr)

            if current_precisions is None or len(current_precisions) == 0:
                print(f"[r_t n={self._n_train}] 提前返回: current_precisions为空", file=sys.stderr)
                return 1.0

            # 首次计算：保存初始精度作为基线
            print(f"[r_t n={self._n_train}] 检查_initial_param_vars: is None={self._initial_param_vars is None}", file=sys.stderr)

            # 【修复】检查核心参数数量是否变化
            # 注意：使用智能过滤后，Ordinal GP的cutpoints变化不会触发此检测
            # 只有真正的模型结构变化（如改变核函数）才会触发
            if self._initial_param_vars is not None and len(self._initial_param_vars) != len(current_precisions):
                print(f"[r_t n={self._n_train}] ⚠️ 核心参数数量变化: {len(self._initial_param_vars)} -> {len(current_precisions)}, 重置基线", file=sys.stderr)
                print(f"[r_t n={self._n_train}] 注意：这通常不应该发生（已过滤掉cutpoints），可能是模型结构改变", file=sys.stderr)
                self._initial_param_vars = None  # 重置基线

            if self._initial_param_vars is None:
                self._initial_param_vars = current_precisions.clone().detach()
                # Debug: 每次都打印初始精度（移除hasattr限制）
                import sys
                print(f"[r_t n={self._n_train}] 首次计算，保存初始精度", file=sys.stderr)
                print(f"[r_t n={self._n_train}] 初始精度: 均值={self._initial_param_vars.mean().item():.6e}, std={self._initial_param_vars.std().item():.6e}, min={self._initial_param_vars.min().item():.6e}, max={self._initial_param_vars.max().item():.6e}", file=sys.stderr)
                return 1.0

            # 计算方差比
            # 因为 Var ≈ 1/precision，所以 r_t = Var_t/Var_0 = prec_0/prec_t
            variance_ratios = self._initial_param_vars / (current_precisions + EPS)
            r_t = variance_ratios.mean().item()

            # Debug: 每次都打印（移除hasattr限制），检测重置
            import sys
            print(f"[r_t n={self._n_train}] 当前精度: 均值={current_precisions.mean().item():.6e}, std={current_precisions.std().item():.6e}", file=sys.stderr)
            print(f"[r_t n={self._n_train}] 初始精度: 均值={self._initial_param_vars.mean().item():.6e} (是否被重置: {id(self._initial_param_vars)})", file=sys.stderr)
            print(f"[r_t n={self._n_train}] 方差比: 均值={variance_ratios.mean().item():.6f}, std={variance_ratios.std().item():.6f}, min={variance_ratios.min().item():.6f}, max={variance_ratios.max().item():.6f}", file=sys.stderr)
            print(f"[r_t n={self._n_train}] r_t={r_t:.6f} (夹值前)", file=sys.stderr)

            # 【诊断增强】打印前10个方差比值，检查是否所有元素都是1.0
            if variance_ratios.shape[0] >= 20:
                print(f"[r_t n={self._n_train}] 方差比前10个: {variance_ratios[:10].tolist()}", file=sys.stderr)
                print(f"[r_t n={self._n_train}] 当前精度前10个: {current_precisions[:10].tolist()}", file=sys.stderr)
                print(f"[r_t n={self._n_train}] 初始精度前10个: {self._initial_param_vars[:10].tolist()}", file=sys.stderr)
            else:
                print(f"[r_t n={self._n_train}] 方差比所有值: {variance_ratios.tolist()}", file=sys.stderr)
                print(f"[r_t n={self._n_train}] 当前精度所有值: {current_precisions.tolist()}", file=sys.stderr)
                print(f"[r_t n={self._n_train}] 初始精度所有值: {self._initial_param_vars.tolist()}", file=sys.stderr)

            # 夹值到[0, 1]
            r_t = float(max(0.0, min(1.0, r_t)))

            return r_t

        except Exception as e:
            import sys
            import traceback
            print(f"[r_t n={self._n_train}] 捕获异常: {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
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

        Returns:
            λ_t ∈ [lambda_min, lambda_max]
        """
        if not self.use_dynamic_lambda:
            return float(self.lambda_max)

        r_t = self.compute_relative_main_variance()

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
