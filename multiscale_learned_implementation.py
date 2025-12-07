"""
多尺度与学习型扰动的实现演示
=================================

这个文件展示如何在现有 EURAnovaPairAcqf 基础上，
实现两个改进方向的具体代码。
"""

import numpy as np
import torch
from typing import Dict, Optional, List, Callable, Tuple
from dataclasses import dataclass


# ============================================================================
# 方向1️⃣：多尺度扰动的实现
# ============================================================================


@dataclass
class MultiScaleConfig:
    """多尺度扰动配置"""

    scales: List[float] = None  # e.g., [0.05, 0.15, 0.3]
    points_per_scale: int = 2  # 每个尺度采样点数

    def __post_init__(self):
        if self.scales is None:
            self.scales = [0.05, 0.15, 0.3]


class MultiScalePerturbationMixin:
    """多尺度扰动混入类 - 添加到 EURAnovaPairAcqf"""

    def __init__(self, multiscale_config: Optional[MultiScaleConfig] = None):
        """初始化多尺度配置"""
        self.multiscale_config = multiscale_config or MultiScaleConfig()
        self.total_local_points = (
            len(self.multiscale_config.scales) * self.multiscale_config.points_per_scale
        )

    def _make_local_multiscale(
        self,
        X_can_t: torch.Tensor,  # (B, d)
        dims: List[int],  # 要扰动的维度
    ) -> torch.Tensor:
        """【改进版】多尺度局部扰动

        返回形状：(B * total_local_points, d)
          其中 total_local_points = len(scales) * points_per_scale

        例子：
          - 如果 scales=[0.05, 0.15, 0.3], points_per_scale=2
          - 则 total_local_points = 3*2 = 6
          - 返回 (B*6, d)
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

        # 为每个尺度创建一个 local perturbation 张量
        all_scale_perturbations = []

        for scale in self.multiscale_config.scales:
            # 该尺度的基础张量 (B, points_per_scale, d)
            base = X_can_t.unsqueeze(1).repeat(
                1, self.multiscale_config.points_per_scale, 1
            )

            # 对指定维度应用扰动
            for k in dims:
                vtype = self.variable_types.get(k) if self.variable_types else None
                sigma = scale * span[k]

                if vtype == "categorical":
                    unique_vals = self._unique_vals_dict.get(k)
                    if unique_vals is not None:
                        samples = np.random.choice(
                            unique_vals,
                            size=(B, self.multiscale_config.points_per_scale),
                        )
                        base[:, :, k] = torch.from_numpy(samples).to(
                            dtype=X_can_t.dtype, device=X_can_t.device
                        )

                elif vtype == "integer":
                    noise = (
                        torch.randn(
                            B,
                            self.multiscale_config.points_per_scale,
                            device=X_can_t.device,
                        )
                        * sigma
                    )
                    base[:, :, k] = torch.round(
                        torch.clamp(base[:, :, k] + noise, min=mn[k], max=mx[k])
                    )

                else:  # continuous
                    noise = (
                        torch.randn(
                            B,
                            self.multiscale_config.points_per_scale,
                            device=X_can_t.device,
                        )
                        * sigma
                    )
                    base[:, :, k] = torch.clamp(
                        base[:, :, k] + noise, min=mn[k], max=mx[k]
                    )

            # 该尺度的所有点 (B, points_per_scale, d)
            all_scale_perturbations.append(base)

        # 拼接所有尺度 (B, total_scales*points_per_scale, d)
        all_perturbs = torch.cat(all_scale_perturbations, dim=1)

        # 展平 (B*total_points, d)
        return all_perturbs.reshape(B * self.total_local_points, d)


# ============================================================================
# 方向2️⃣：学习型扰动分布的实现
# ============================================================================


@dataclass
class DimensionLearningStats:
    """单个维度的学习统计信息"""

    learning_rate: float = 0.5  # 参数收敛速率 [0, 1]
    interaction_freq: int = 0  # 参与重要交互的频数
    residual_correlation: float = 0.0  # 该维与残差的相关性
    residual_kurtosis: float = 3.0  # 残差的尾部厚度


class LearnedPerturbationAdaptor:
    """学习型扰动适配器"""

    def __init__(self, n_dims: int):
        self.n_dims = n_dims
        self.dim_stats: Dict[int, DimensionLearningStats] = {
            i: DimensionLearningStats() for i in range(n_dims)
        }
        self.interaction_pairs = []
        self.effect_magnitudes: Dict[Tuple[int, int], float] = {}

    def update_from_training(
        self,
        model,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        interaction_pairs: List[Tuple[int, int]],
        effect_magnitudes: Dict[Tuple[int, int], float],
    ):
        """从训练数据和当前模型更新统计信息

        Args:
            model: 当前 GP 模型
            X_train: 训练特征 (n, d)
            y_train: 训练目标 (n,)
            interaction_pairs: 交互对列表
            effect_magnitudes: 每个交互对的效应值 {(i,j): magnitude}
        """

        # ===== 信号1：参数收敛速率 =====
        initial_vars = self._extract_initial_vars(model)
        current_vars = self._extract_current_vars(model)

        for dim_idx in range(self.n_dims):
            if initial_vars is None or current_vars is None:
                learning_rate = 0.5
            else:
                # 方差缩减的比例
                reduction = (initial_vars[dim_idx] - current_vars[dim_idx]) / (
                    initial_vars[dim_idx] + 1e-8
                )
                learning_rate = float(np.clip(reduction, 0.0, 1.0))

            self.dim_stats[dim_idx].learning_rate = learning_rate

        # ===== 信号2：交互活跃度 =====
        self.interaction_pairs = interaction_pairs
        self.effect_magnitudes = effect_magnitudes

        # 重置交互计数
        for i in range(self.n_dims):
            self.dim_stats[i].interaction_freq = 0

        # 统计：高效应的交互中每维出现的次数
        effect_threshold = np.median(list(effect_magnitudes.values())) * 1.5
        for (i, j), magnitude in effect_magnitudes.items():
            if magnitude > effect_threshold:
                self.dim_stats[i].interaction_freq += 1
                self.dim_stats[j].interaction_freq += 1

        # ===== 信号3：残差统计 =====
        try:
            with torch.no_grad():
                posterior = model.posterior(X_train)
                mean = posterior.mean.squeeze(-1)
                var = (
                    posterior.variance.squeeze(-1)
                    if hasattr(posterior, "variance")
                    else torch.ones_like(mean)
                )

            std = torch.sqrt(torch.clamp(var, min=1e-8))
            residuals = (y_train - mean) / std
            residuals_np = residuals.detach().cpu().numpy()

            X_train_np = X_train.detach().cpu().numpy()

            for dim_idx in range(self.n_dims):
                # 相关性
                try:
                    corr = np.corrcoef(X_train_np[:, dim_idx], residuals_np)[0, 1]
                except Exception:
                    corr = 0.0

                # 尾部厚度（峰度）
                try:
                    from scipy.stats import kurtosis

                    kurt = float(kurtosis(residuals_np))
                except Exception:
                    kurt = 3.0

                self.dim_stats[dim_idx].residual_correlation = float(
                    np.clip(corr, -1, 1)
                )
                self.dim_stats[dim_idx].residual_kurtosis = kurt

        except Exception:
            pass  # 如果计算失败，保持默认值

    def get_perturbation_sampler(self, dimension: int) -> Callable:
        """为指定维度获取定制的扰动采样函数

        Returns:
            sampler: Callable[[tuple], torch.Tensor]
              输入 size=(B, N)，输出 (B, N) 的噪声张量
        """

        stats = self.dim_stats.get(dimension)
        if stats is None:
            # 后备方案：简单高斯
            return lambda size: torch.randn(size) * 0.1

        lr = stats.learning_rate
        freq = stats.interaction_freq
        corr = stats.residual_correlation
        kurt = stats.residual_kurtosis

        # ===== 基于学习率的分布选择 =====
        if lr > 0.7:  # 参数已充分学习
            # 用窄分布，标准差与学习进度反向（更小）
            std_base = (1 - lr) * 0.05 + 0.01  # [0.01, 0.05]

            def sampler(size):
                return torch.randn(size) * std_base

        elif lr < 0.3:  # 参数学习不足
            # 用宽分布 + 厚尾，方便探索
            std_base = 0.25

            def sampler(size):
                # 混合：90% 标准高斯 + 10% 学生t（厚尾，模拟极值）
                noise_normal = torch.randn(size) * std_base

                # 学生t 分布：df=3（比高斯有更厚的尾）
                noise_t = (
                    torch.randn(size)
                    / torch.sqrt(torch.chi2.sample((3,)) / 3)
                    * std_base
                    * 1.5
                )

                mask = torch.rand(size) < 0.9
                return torch.where(mask, noise_normal, noise_t)

        else:  # 中等学习进度
            std_base = 0.15

            def sampler(size):
                return torch.randn(size) * std_base

        # ===== 基于交互频数的修正 =====
        original_sampler = sampler

        if freq >= 3:  # 高频率参与交互

            def sampler(size):
                noise = original_sampler(size)
                # 20% 概率增大幅度（偶尔跳远，帮助探索交互边界）
                mask = torch.rand(size) < 0.2
                return torch.where(mask, noise * 2.5, noise)

        # ===== 基于残差相关性的微调 =====
        # 如果该维与残差高度相关，说明该维的预测效果差
        # → 增加探索幅度
        if abs(corr) > 0.3:
            original_sampler_2 = sampler

            def sampler(size):
                noise = original_sampler_2(size)
                # 增加 20% 的标准差
                return noise * 1.2

        return sampler

    def _extract_initial_vars(self, model) -> Optional[np.ndarray]:
        """提取初始参数方差（需要缓存）"""
        # 这通常应在首次调用时缓存
        if not hasattr(self, "_initial_param_vars"):
            self._initial_param_vars = self._compute_param_vars(model)
        return self._initial_param_vars

    def _extract_current_vars(self, model) -> Optional[np.ndarray]:
        """提取当前参数方差"""
        return self._compute_param_vars(model)

    @staticmethod
    def _compute_param_vars(model) -> Optional[np.ndarray]:
        """计算模型参数方差（Laplace 近似）"""
        try:
            # 这是一个简化版本，实际应该调用模型的 Hessian 计算
            # 在完整实现中，应该重用 EURAnovaPairAcqf._extract_parameter_variances_laplace()
            return None
        except Exception:
            return None


# ============================================================================
# 集成示例：改进后的采集函数骨架
# ============================================================================


class EURAnovaPairAcqfEnhanced(MultiScalePerturbationMixin):
    """增强版的 EURAnovaPairAcqf，集成多尺度和学习型扰动"""

    def __init__(
        self,
        model,
        # 原有参数...
        # 多尺度参数
        use_multiscale: bool = False,
        multiscale_config: Optional[MultiScaleConfig] = None,
        # 学习型扰动参数
        use_learned_perturbation: bool = False,
        # 其他...
    ):
        super().__init__(multiscale_config)
        self.model = model
        self.use_multiscale = use_multiscale
        self.use_learned_perturbation = use_learned_perturbation

        if use_learned_perturbation:
            # 假设 d 维已知
            d = (
                getattr(model, "train_inputs", [[None]])[0].shape[1]
                if (hasattr(model, "train_inputs") and model.train_inputs is not None)
                else 10
            )  # 默认值
            self.learned_adaptor = LearnedPerturbationAdaptor(d)
        else:
            self.learned_adaptor = None

    def _make_local_hybrid_improved(
        self,
        X_can_t: torch.Tensor,
        dims: List[int],
    ) -> torch.Tensor:
        """改进版的局部扰动，支持多尺度和学习型"""

        if self.use_multiscale:
            return self._make_local_multiscale(X_can_t, dims)

        # 否则使用原始单尺度方法
        # ... 原始 _make_local_hybrid 逻辑
        pass

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """前向传播（集成改进）"""
        # 更新学习统计（如果启用）
        if self.use_learned_perturbation and self.learned_adaptor is not None:
            self.learned_adaptor.update_from_training(
                self.model,
                X_train=self.model.train_inputs[0],
                y_train=self.model.train_targets,
                interaction_pairs=self._pairs,
                effect_magnitudes={},  # 从前次评估获取
            )

        # 其余逻辑与原始相同，但调用 _make_local_hybrid_improved
        # ...
        pass


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    print("多尺度与学习型扰动的实现框架已准备就绪")
    print()
    print("用法1：启用多尺度")
    print("-------")
    print(
        """
    multiscale_cfg = MultiScaleConfig(
        scales=[0.05, 0.15, 0.3],
        points_per_scale=2
    )
    acqf = EURAnovaPairAcqfEnhanced(
        model=model,
        use_multiscale=True,
        multiscale_config=multiscale_cfg,
    )
    """
    )

    print()
    print("用法2：启用学习型扰动")
    print("-------")
    print(
        """
    acqf = EURAnovaPairAcqfEnhanced(
        model=model,
        use_learned_perturbation=True,
    )
    # 在 forward() 时自动更新学习统计
    """
    )

    print()
    print("用法3：同时启用两者")
    print("-------")
    print(
        """
    acqf = EURAnovaPairAcqfEnhanced(
        model=model,
        use_multiscale=True,
        use_learned_perturbation=True,
        multiscale_config=MultiScaleConfig(
            scales=[0.05, 0.15],  # 减少尺度（因为学习分布已提供多样性）
            points_per_scale=2
        )
    )
    """
    )
