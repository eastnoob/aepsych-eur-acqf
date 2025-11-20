"""
预热策略实现示例：混合策略（策略C）

适用场景：
- 总被试：30人
- 每人样本：25-30次
- 总样本：750-900
- 预热样本：150（5-6人）
- 因子数：6个
"""

import numpy as np
from scipy.spatial.distance import cdist
from itertools import combinations


class WarmupDesigner:
    """预热阶段设计生成器"""

    def __init__(self, n_factors: int, factor_ranges: list):
        """
        Args:
            n_factors: 因子数量
            factor_ranges: 每个因子的取值范围 [(min, max), ...]
        """
        self.n_factors = n_factors
        self.factor_ranges = np.array(factor_ranges)

    def generate_strategy_a(self, n_samples: int = 150) -> np.ndarray:
        """策略A：纯Space-filling（推荐基线）

        Args:
            n_samples: 预热样本数

        Returns:
            X_warmup: (n_samples, n_factors)
        """
        print(f"[策略A] 生成 {n_samples} 个Space-filling样本...")
        return self._maximin_lhs(n_samples)

    def generate_strategy_c(
        self,
        n_samples_main: int = 120,
        n_samples_interaction: int = 30,
        priority_pairs: list = None
    ) -> tuple:
        """策略C：混合预热（有先验知识时使用）

        Args:
            n_samples_main: 主效应预热样本数
            n_samples_interaction: 交互效应预热样本数
            priority_pairs: 优先采样的交互对，例如 [(0,1), (2,3)]
                           如果为None，则选择边界覆盖最差的交互对

        Returns:
            X_main: (n_samples_main, n_factors) 主效应样本
            X_interaction: (n_samples_interaction, n_factors) 交互样本
            selected_pairs: 实际采样的交互对
        """
        print(f"[策略C] 阶段1a: 生成 {n_samples_main} 个主效应样本...")
        X_main = self._maximin_lhs(n_samples_main)

        # 确定要采样的交互对
        if priority_pairs is None:
            # 自动选择：找到被主效应样本覆盖最差的交互对
            priority_pairs = self._select_undersampled_pairs(X_main, n_pairs=2)

        print(f"[策略C] 阶段1b: 针对交互对 {priority_pairs} 生成 {n_samples_interaction} 个样本...")
        X_interaction = self._sample_interaction_pairs(
            priority_pairs,
            n_samples_per_pair=n_samples_interaction // len(priority_pairs)
        )

        return X_main, X_interaction, priority_pairs

    def _maximin_lhs(self, n_samples: int, n_candidates: int = 1000) -> np.ndarray:
        """Maximin Latin Hypercube Sampling

        策略：生成多个LHS候选，选择最小成对距离最大的那个
        """
        best_X = None
        best_min_dist = -np.inf

        for _ in range(10):  # 生成10个候选
            # 标准LHS
            X_uniform = np.random.rand(n_samples, self.n_factors)
            for j in range(self.n_factors):
                order = np.random.permutation(n_samples)
                X_uniform[:, j] = (order + X_uniform[:, j]) / n_samples

            # 映射到实际范围
            X = X_uniform * (self.factor_ranges[:, 1] - self.factor_ranges[:, 0]) + self.factor_ranges[:, 0]

            # 计算最小成对距离
            dists = cdist(X, X)
            np.fill_diagonal(dists, np.inf)
            min_dist = dists.min()

            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_X = X

        print(f"  ✓ 最小成对距离: {best_min_dist:.4f}")
        return best_X

    def _select_undersampled_pairs(self, X_main: np.ndarray, n_pairs: int = 2) -> list:
        """选择被主效应样本覆盖最差的交互对

        策略：对于每个交互对(i,j)，计算其4个极端点组合在X_main中的最近距离
              选择总距离最大的n_pairs个交互对
        """
        all_pairs = list(combinations(range(self.n_factors), 2))
        pair_scores = []

        for i, j in all_pairs:
            # 4个极端点：(min,min), (min,max), (max,min), (max,max)
            corners = np.array([
                [self.factor_ranges[i, 0], self.factor_ranges[j, 0]],
                [self.factor_ranges[i, 0], self.factor_ranges[j, 1]],
                [self.factor_ranges[i, 1], self.factor_ranges[j, 0]],
                [self.factor_ranges[i, 1], self.factor_ranges[j, 1]],
            ])

            # 计算每个极端点到X_main中对应维度的最近距离
            X_ij = X_main[:, [i, j]]
            dists = cdist(corners, X_ij).min(axis=1)

            # 总距离作为"覆盖不足"的指标
            pair_scores.append(((i, j), dists.sum()))

        # 选择得分最高的n_pairs个
        pair_scores.sort(key=lambda x: x[1], reverse=True)
        selected = [pair for pair, score in pair_scores[:n_pairs]]

        print(f"  ✓ 自动选择覆盖不足的交互对: {selected}")
        return selected

    def _sample_interaction_pairs(
        self,
        pairs: list,
        n_samples_per_pair: int
    ) -> np.ndarray:
        """为指定交互对生成正交对比样本

        策略：2^k 完全因子设计 + 中心点 + 轴向点（类似CCD）
        """
        X_list = []

        for i, j in pairs:
            # 2^2 完全因子设计（4个角点）
            levels_i = [self.factor_ranges[i, 0], self.factor_ranges[i, 1]]
            levels_j = [self.factor_ranges[j, 0], self.factor_ranges[j, 1]]

            factorial_ij = np.array([
                [li, lj]
                for li in levels_i
                for lj in levels_j
            ])  # (4, 2)

            # 添加中心点
            center_ij = np.array([[
                np.mean(levels_i),
                np.mean(levels_j)
            ]])  # (1, 2)

            # 添加轴向点（单因子变化）
            axial_ij = np.array([
                [levels_i[0], np.mean(levels_j)],
                [levels_i[1], np.mean(levels_j)],
                [np.mean(levels_i), levels_j[0]],
                [np.mean(levels_i), levels_j[1]],
            ])  # (4, 2)

            # 合并（4+1+4=9个点）
            design_ij = np.vstack([factorial_ij, center_ij, axial_ij])  # (9, 2)

            # 其他维度：使用随机值（从主效应LHS中采样）
            n_design = design_ij.shape[0]
            X_pair = np.random.rand(n_design, self.n_factors)
            for k in range(self.n_factors):
                X_pair[:, k] = X_pair[:, k] * (self.factor_ranges[k, 1] - self.factor_ranges[k, 0]) + self.factor_ranges[k, 0]

            # 替换维度i和j
            X_pair[:, i] = design_ij[:, 0]
            X_pair[:, j] = design_ij[:, 1]

            X_list.append(X_pair)

        return np.vstack(X_list)


# ========== 使用示例 ==========

def example_usage():
    """实际使用示例"""

    # 假设6个因子，每个因子范围[0, 1]（标准化后）
    n_factors = 6
    factor_ranges = [(0, 1)] * n_factors

    designer = WarmupDesigner(n_factors, factor_ranges)

    # ===== 策略A：纯Space-filling（推荐基线）=====
    print("\n" + "="*60)
    print("策略A：纯Space-filling")
    print("="*60)
    X_warmup_a = designer.generate_strategy_a(n_samples=150)
    print(f"生成样本形状: {X_warmup_a.shape}")

    # ===== 策略C：混合预热（有先验时使用）=====
    print("\n" + "="*60)
    print("策略C：混合预热")
    print("="*60)

    # 情况1：有先验知识（例如文献报告因子0和1有交互）
    X_main, X_int, pairs = designer.generate_strategy_c(
        n_samples_main=120,
        n_samples_interaction=30,
        priority_pairs=[(0, 1), (2, 5)]  # 先验指定
    )
    print(f"主效应样本: {X_main.shape}")
    print(f"交互样本: {X_int.shape}")
    print(f"采样的交互对: {pairs}")

    # 情况2：无先验知识（自动选择覆盖不足的交互对）
    print("\n" + "-"*60)
    X_main2, X_int2, pairs2 = designer.generate_strategy_c(
        n_samples_main=120,
        n_samples_interaction=30,
        priority_pairs=None  # 自动选择
    )
    print(f"主效应样本: {X_main2.shape}")
    print(f"交互样本: {X_int2.shape}")
    print(f"自动选择的交互对: {pairs2}")

    # ===== 验证覆盖质量 =====
    print("\n" + "="*60)
    print("覆盖质量对比")
    print("="*60)

    # 策略A
    dists_a = cdist(X_warmup_a, X_warmup_a)
    np.fill_diagonal(dists_a, np.inf)
    print(f"策略A 最小成对距离: {dists_a.min():.4f}")

    # 策略C（合并主效应+交互）
    X_warmup_c = np.vstack([X_main2, X_int2])
    dists_c = cdist(X_warmup_c, X_warmup_c)
    np.fill_diagonal(dists_c, np.inf)
    print(f"策略C 最小成对距离: {dists_c.min():.4f}")

    return X_warmup_a, X_warmup_c


if __name__ == "__main__":
    np.random.seed(42)
    X_a, X_c = example_usage()

    print("\n" + "="*60)
    print("✅ 推荐使用策略A（纯Space-filling）")
    print("="*60)
    print("理由：")
    print("1. 您的采集函数有动态权重机制（λ_t），会自动发现交互")
    print("2. 预热阶段重点是为主效应提供良好估计（交互检测的基础）")
    print("3. 不确定性导向策略会自然探索被遗漏的交互区域")
    print("\n如果有明确的先验知识（文献/pilot study），可使用策略C")
    print("="*60)
