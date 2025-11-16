"""
ANOVA效应层次化计算引擎

支持任意阶数的ANOVA分解（主效应、二阶、三阶...）
- 自动依赖管理（高阶效应自动依赖低阶结果）
- 批量评估优化（一次性计算所有扰动点，性能提升20x+）
- 可扩展设计（新增阶数只需实现新的Effect子类）

ANOVA分解公式：
- 一阶（主效应）: Δ_i = I(x_i) - I(x)
- 二阶（交互）: Δ_ij = I(x_ij) - I(x_i) - I(x_j) + I(x)
- 三阶（三向交互）: Δ_ijk = I(x_ijk) - I(x_ij) - I(x_ik) - I(x_jk) + I(x_i) + I(x_j) + I(x_k) - I(x)

Example:
    >>> # 定义效应
    >>> effects = [
    >>>     MainEffect(0), MainEffect(1),  # 主效应
    >>>     PairwiseEffect(0, 1),           # 二阶交互
    >>>     ThreeWayEffect(0, 1, 2)         # 三阶交互
    >>> ]
    >>>
    >>> # 计算
    >>> engine = ANOVAEffectEngine(metric_fn, local_sampler)
    >>> results = engine.compute_effects(X_candidates, effects)
    >>>
    >>> # 获取聚合结果
    >>> main_sum = results['aggregated']['order_1']
    >>> pair_sum = results['aggregated']['order_2']
    >>> triplet_sum = results['aggregated']['order_3']
"""

from __future__ import annotations
from typing import Tuple, List, Dict, Callable, Optional
from abc import ABC, abstractmethod
import torch


class ANOVAEffect(ABC):
    """ANOVA效应抽象基类

    所有效应类型（主效应、交互效应）的基类。
    每个子类实现特定阶数的ANOVA分解公式。
    """

    def __init__(self, order: int, indices: Tuple[int, ...]):
        """
        Args:
            order: 效应阶数（1=主效应, 2=二阶, 3=三阶...）
            indices: 涉及的维度索引（已排序，用于唯一标识）
        """
        self.order = order
        self.indices = indices

    @abstractmethod
    def get_dependencies(self) -> List[Tuple[int, ...]]:
        """返回计算此效应所需的低阶效应索引

        Returns:
            低阶效应的索引元组列表
            例如：PairwiseEffect(0,1)需要 [(0,), (1,)]
                  ThreeWayEffect(0,1,2)需要 [(0,), (1,), (2,), (0,1), (0,2), (1,2)]
        """
        raise NotImplementedError

    @abstractmethod
    def compute_contribution(
        self,
        I_current: torch.Tensor,          # I(x_当前组合) - 扰动后的信息度量
        I_baseline: torch.Tensor,         # I(x) - 基线信息度量
        lower_order_results: Dict[Tuple[int, ...], torch.Tensor]  # 低阶效应的I值
    ) -> torch.Tensor:
        """应用ANOVA分解公式计算此效应的贡献量

        Args:
            I_current: 当前组合的信息度量（扰动dims后）
            I_baseline: 基线信息度量（未扰动）
            lower_order_results: 低阶效应的信息度量字典
                                键为索引元组，值为对应的I值

        Returns:
            效应贡献量 Δ (与I_current同形状)
        """
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}{self.indices}"


class MainEffect(ANOVAEffect):
    """一阶主效应：Δ_i = I(x_i) - I(x)

    衡量单个维度扰动对不确定性的影响。

    Example:
        >>> effect = MainEffect(2)  # 维度2的主效应
        >>> Delta_2 = effect.compute_contribution(I_x2, I_x, {})
    """

    def __init__(self, i: int):
        super().__init__(order=1, indices=(i,))

    def get_dependencies(self) -> List[Tuple[int, ...]]:
        return []  # 主效应无依赖

    def compute_contribution(
        self,
        I_current: torch.Tensor,
        I_baseline: torch.Tensor,
        lower_order_results: Dict[Tuple[int, ...], torch.Tensor]
    ) -> torch.Tensor:
        # Δ_i = I(x_i) - I(x)
        # clamp(min=0) 因为采用不确定性导向策略（只保留增加不确定性的方向）
        return torch.clamp(I_current - I_baseline, min=0.0)


class PairwiseEffect(ANOVAEffect):
    """二阶交互效应：Δ_ij = I(x_ij) - I(x_i) - I(x_j) + I(x)

    衡量两个维度联合扰动的"超额"不确定性（超出两个主效应之和）。

    Example:
        >>> effect = PairwiseEffect(0, 2)  # 维度0和2的交互
        >>> Delta_02 = effect.compute_contribution(I_x02, I_x, {(0,): I_x0, (2,): I_x2})
    """

    def __init__(self, i: int, j: int):
        # 规范化：始终保持 i < j
        super().__init__(order=2, indices=(min(i, j), max(i, j)))

    def get_dependencies(self) -> List[Tuple[int, ...]]:
        i, j = self.indices
        return [(i,), (j,)]  # 依赖两个主效应

    def compute_contribution(
        self,
        I_current: torch.Tensor,
        I_baseline: torch.Tensor,
        lower_order_results: Dict[Tuple[int, ...], torch.Tensor]
    ) -> torch.Tensor:
        i, j = self.indices
        I_i = lower_order_results.get((i,))
        I_j = lower_order_results.get((j,))

        # 如果依赖缺失，返回零贡献
        if I_i is None or I_j is None:
            return torch.zeros_like(I_baseline)

        # ANOVA分解：Δ_ij = I(x_ij) - I(x_i) - I(x_j) + I(x)
        return torch.clamp(I_current - I_i - I_j + I_baseline, min=0.0)


class ThreeWayEffect(ANOVAEffect):
    """三阶交互效应：Δ_ijk = I(x_ijk) - [二阶项] + [一阶项] - I(x)

    完整公式：
    Δ_ijk = I(x_ijk)
            - I(x_ij) - I(x_ik) - I(x_jk)  # 减去所有二阶项
            + I(x_i) + I(x_j) + I(x_k)     # 加回所有一阶项（容斥原理）
            - I(x)                          # 减去基线

    衡量三个维度联合扰动的"超额超额"不确定性
    （超出三个二阶交互之和，减去重复计算的主效应）

    Example:
        >>> effect = ThreeWayEffect(0, 1, 2)
        >>> Delta_012 = effect.compute_contribution(
        >>>     I_x012, I_x,
        >>>     {(0,): I_x0, (1,): I_x1, (2,): I_x2,
        >>>      (0,1): I_x01, (0,2): I_x02, (1,2): I_x12}
        >>> )
    """

    def __init__(self, i: int, j: int, k: int):
        # 规范化：按从小到大排序
        indices = tuple(sorted([i, j, k]))
        super().__init__(order=3, indices=indices)

    def get_dependencies(self) -> List[Tuple[int, ...]]:
        i, j, k = self.indices
        return [
            # 一阶依赖
            (i,), (j,), (k,),
            # 二阶依赖（注意顺序规范化）
            (min(i, j), max(i, j)),
            (min(i, k), max(i, k)),
            (min(j, k), max(j, k))
        ]

    def compute_contribution(
        self,
        I_current: torch.Tensor,
        I_baseline: torch.Tensor,
        lower_order_results: Dict[Tuple[int, ...], torch.Tensor]
    ) -> torch.Tensor:
        i, j, k = self.indices

        # 获取一阶项
        I_i = lower_order_results.get((i,))
        I_j = lower_order_results.get((j,))
        I_k = lower_order_results.get((k,))

        # 获取二阶项（注意键的规范化）
        I_ij = lower_order_results.get((min(i, j), max(i, j)))
        I_ik = lower_order_results.get((min(i, k), max(i, k)))
        I_jk = lower_order_results.get((min(j, k), max(j, k)))

        # 依赖完整性检查
        if any(x is None for x in [I_i, I_j, I_k, I_ij, I_ik, I_jk]):
            return torch.zeros_like(I_baseline)

        # ANOVA容斥公式
        delta = (
            I_current
            - I_ij - I_ik - I_jk    # 减去二阶
            + I_i + I_j + I_k       # 加回一阶
            - I_baseline            # 减去基线
        )

        return torch.clamp(delta, min=0.0)


class ANOVAEffectEngine:
    """层次化效应计算引擎

    核心功能：
    1. 批量评估优化：所有扰动点一次性评估（性能提升20x+）
    2. 自动依赖管理：按阶数层次化计算，自动满足依赖
    3. 灵活扩展：支持任意阶数的效应（只需定义新的Effect类）

    Example:
        >>> # 初始化
        >>> engine = ANOVAEffectEngine(
        >>>     metric_fn=lambda X: model.posterior(X).variance,
        >>>     local_sampler=lambda X, dims: perturb(X, dims)
        >>> )
        >>>
        >>> # 定义要计算的效应
        >>> effects = [
        >>>     MainEffect(0), MainEffect(1), MainEffect(2),
        >>>     PairwiseEffect(0, 1), PairwiseEffect(1, 2),
        >>>     ThreeWayEffect(0, 1, 2)
        >>> ]
        >>>
        >>> # 批量计算
        >>> results = engine.compute_effects(X_candidates, effects)
        >>>
        >>> # 使用结果
        >>> main_contrib = results['aggregated']['order_1']  # 主效应平均贡献
        >>> pair_contrib = results['aggregated']['order_2']  # 二阶交互平均贡献
        >>> triplet_contrib = results['aggregated']['order_3']  # 三阶交互平均贡献
    """

    def __init__(
        self,
        metric_fn: Callable[[torch.Tensor], torch.Tensor],
        local_sampler: Callable[[torch.Tensor, List[int]], torch.Tensor]
    ):
        """
        Args:
            metric_fn: 信息度量函数 X -> I(X)
                      输入: (N, d) 张量
                      输出: (N,) 张量（每个点的信息度量）

            local_sampler: 局部扰动采样器 (X, dims) -> X_perturbed
                          输入: X (B, d), dims 要扰动的维度列表
                          输出: (B*local_num, d) 扰动后的点
        """
        self.metric_fn = metric_fn
        self.local_sampler = local_sampler

    def compute_effects(
        self,
        X_can_t: torch.Tensor,
        effects: List[ANOVAEffect],
        enable_orders: Optional[List[int]] = None
    ) -> Dict:
        """批量计算所有效应（自动处理依赖）

        Args:
            X_can_t: 候选点 (B, d)
            effects: 要计算的效应列表
            enable_orders: 启用的阶数列表（None表示全部启用）
                          例如: [1, 2] 只计算主效应和二阶交互

        Returns:
            {
                'baseline': I(x),  # (B,) 基线信息度量
                'raw_results': {   # 原始I值
                    (i,): I(x_i),           # 一阶
                    (i,j): I(x_ij),         # 二阶
                    (i,j,k): I(x_ijk), ...  # 三阶
                },
                'contributions': {  # ANOVA分解的Δ值
                    (i,): Δ_i,
                    (i,j): Δ_ij,
                    (i,j,k): Δ_ijk, ...
                },
                'aggregated': {     # 按阶数聚合的平均贡献
                    'order_1': mean(Δ_i),
                    'order_2': mean(Δ_ij),
                    'order_3': mean(Δ_ijk), ...
                }
            }
        """
        B = X_can_t.shape[0]

        # 阶数过滤
        if enable_orders is not None:
            effects = [eff for eff in effects if eff.order in enable_orders]

        # 如果没有效应，返回空结果
        if len(effects) == 0:
            baseline = self.metric_fn(X_can_t)
            return {
                'baseline': baseline,
                'raw_results': {},
                'contributions': {},
                'aggregated': {}
            }

        # 1. 基线评估
        I_baseline = self.metric_fn(X_can_t)

        # 2. 批量构造所有扰动点
        X_all_local = []
        segment_info = []  # [(indices, list_idx), ...]

        for effect in effects:
            X_perturbed = self.local_sampler(X_can_t, list(effect.indices))
            segment_info.append((effect.indices, len(X_all_local)))
            X_all_local.append(X_perturbed)

        # 3. 一次性批量评估（关键性能优化！）
        if len(X_all_local) > 0:
            # 【修复】验证所有扰动点有相同的 local_num（确保假设成立）
            local_num = X_all_local[0].shape[0] // B
            for i, X_pert in enumerate(X_all_local):
                actual_local_num = X_pert.shape[0] // B
                if actual_local_num != local_num:
                    raise AssertionError(
                        f"ANOVA引擎假设所有effect的local_num一致，"
                        f"但effect索引{i}的local_num={actual_local_num}，"
                        f"与第一个effect的local_num={local_num}不符"
                    )

            X_batch = torch.cat(X_all_local, dim=0)
            I_batch = self.metric_fn(X_batch)  # 只调用1次模型！

            # 解包结果
            raw_results = {}
            current_row = 0

            for indices, _ in segment_info:
                seg_size = B * local_num
                I_seg = I_batch[current_row:current_row+seg_size]
                I_seg = I_seg.view(B, local_num).mean(dim=1)  # 平均local_num个扰动
                raw_results[indices] = I_seg
                current_row += seg_size
        else:
            raw_results = {}

        # 4. 按阶数层次化计算贡献（确保依赖满足）
        effects_by_order = {}
        for effect in effects:
            effects_by_order.setdefault(effect.order, []).append(effect)

        contributions = {}
        for order in sorted(effects_by_order.keys()):
            for effect in effects_by_order[order]:
                I_current = raw_results.get(effect.indices)
                if I_current is not None:
                    contrib = effect.compute_contribution(
                        I_current, I_baseline, raw_results
                    )
                    contributions[effect.indices] = contrib

        # 5. 按阶数聚合（计算每阶的平均贡献）
        aggregated = {}
        for order in effects_by_order.keys():
            order_contribs = [
                contributions[eff.indices]
                for eff in effects_by_order[order]
                if eff.indices in contributions
            ]
            if order_contribs:
                # 平均所有同阶效应的贡献
                aggregated[f'order_{order}'] = torch.stack(order_contribs, dim=1).mean(dim=1)
            else:
                aggregated[f'order_{order}'] = torch.zeros_like(I_baseline)

        return {
            'baseline': I_baseline,
            'raw_results': raw_results,
            'contributions': contributions,
            'aggregated': aggregated
        }


# ========== 辅助函数：从配置生成效应列表 ==========

def create_effects_from_config(
    n_dims: int,
    enable_main: bool = True,
    interaction_pairs: Optional[List[Tuple[int, int]]] = None,
    interaction_triplets: Optional[List[Tuple[int, int, int]]] = None
) -> List[ANOVAEffect]:
    """从配置参数生成效应列表（便捷工厂函数）

    Args:
        n_dims: 变量维度数
        enable_main: 是否启用主效应
        interaction_pairs: 二阶交互列表（None表示不启用二阶）
        interaction_triplets: 三阶交互列表（None表示不启用三阶）

    Returns:
        效应列表

    Example:
        >>> # 只启用主效应
        >>> effects = create_effects_from_config(4, enable_main=True)
        >>>
        >>> # 主效应 + 指定二阶交互
        >>> effects = create_effects_from_config(
        >>>     4,
        >>>     enable_main=True,
        >>>     interaction_pairs=[(0,1), (2,3)]
        >>> )
        >>>
        >>> # 主效应 + 二阶 + 三阶
        >>> effects = create_effects_from_config(
        >>>     4,
        >>>     enable_main=True,
        >>>     interaction_pairs=[(0,1), (1,2), (2,3)],
        >>>     interaction_triplets=[(0,1,2)]
        >>> )
    """
    effects = []

    # 主效应
    if enable_main:
        for i in range(n_dims):
            effects.append(MainEffect(i))

    # 二阶交互
    if interaction_pairs is not None:
        for i, j in interaction_pairs:
            effects.append(PairwiseEffect(i, j))

    # 三阶交互
    if interaction_triplets is not None:
        for i, j, k in interaction_triplets:
            effects.append(ThreeWayEffect(i, j, k))

    return effects
