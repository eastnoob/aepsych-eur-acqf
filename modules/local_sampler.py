"""
混合变量类型局部扰动生成器

支持三种变量类型的局部探索：
- 分类变量：从历史观测值离散采样
- 整数变量：高斯扰动后舍入
- 连续变量：标准高斯扰动

Example:
    >>> sampler = LocalSampler(
    >>>     variable_types={0: 'categorical', 2: 'integer'},
    >>>     local_jitter_frac=0.1,
    >>>     local_num=4
    >>> )
    >>> sampler.update_data(X_train_np)
    >>> X_perturbed = sampler.sample(X_candidates, dims=[0, 2])
"""

from __future__ import annotations
from typing import Optional, Dict, List, Sequence
import warnings
import numpy as np
import torch


class LocalSampler:
    """混合变量类型局部扰动采样器

    核心功能：
    - 自动识别变量类型
    - 分类维：100%合法值（从历史采样）
    - 整数维：舍入+夹值
    - 连续维：高斯扰动
    """

    def __init__(
        self,
        variable_types: Optional[Dict[int, str]] = None,
        local_jitter_frac: float = 0.1,
        local_num: int = 4,
        random_seed: Optional[int] = 42,
        # ========== 混合扰动策略参数 ==========
        use_hybrid_perturbation: bool = False,
        exhaustive_level_threshold: int = 3,
        exhaustive_use_cyclic_fill: bool = True,
        # ========== 自动计算 local_num ==========
        auto_compute_local_num: bool = False,
        auto_local_num_max: int = 12,
    ):
        """
        Args:
            variable_types: 变量类型字典 {dim_idx: type_str}
                           type_str ∈ {'categorical', 'integer', 'continuous'}
            local_jitter_frac: 扰动幅度（相对特征范围）
            local_num: 每个候选点生成的扰动数（手动设置时使用）
            random_seed: 随机种子（None表示不固定）
            use_hybrid_perturbation: 是否启用混合扰动策略（默认False，向后兼容）
            exhaustive_level_threshold: 对多少水平以下的离散变量使用穷举（默认3）
                                       例如：threshold=3表示2-3水平变量用穷举，≥4水平用高斯
            exhaustive_use_cyclic_fill: 穷举时是否循环填充到local_num（默认True）
                                       True:  3水平+local_num=4 → [0,1,2,0]
                                       False: 3水平+local_num=4 → [0,1,2]
            auto_compute_local_num: 是否自动计算local_num（默认False，手动配置）
                                   True: 根据低水平离散变量自动计算LCM
                                   False: 使用手动设置的local_num值
            auto_local_num_max: 自动计算local_num时的上限（默认12，避免成本爆炸）
        """
        self.variable_types = variable_types
        self.local_jitter_frac = local_jitter_frac
        self._local_num_manual = int(local_num)  # 保存手动设置值
        self.random_seed = random_seed

        # 混合扰动策略参数
        self.use_hybrid_perturbation = use_hybrid_perturbation
        self.exhaustive_level_threshold = int(exhaustive_level_threshold)
        self.exhaustive_use_cyclic_fill = exhaustive_use_cyclic_fill

        # 自动计算参数
        self.auto_compute_local_num = auto_compute_local_num
        self.auto_local_num_max = int(auto_local_num_max)

        # local_num 初始化（可能在 update_data 中被自动计算覆盖）
        self.local_num = self._local_num_manual

        # 【修复】使用实例级 RNG（避免全局污染）
        # 参考：https://numpy.org/doc/stable/reference/random/generator.html
        if random_seed is not None:
            # Convert to int to handle config parsing floats (e.g., 42.0 -> 42)
            random_seed_int = int(random_seed)
            self._np_rng = np.random.default_rng(random_seed_int)
            # 同时设置 torch 随机种子（保证跨框架可复现）
            torch.manual_seed(random_seed_int)
        else:
            self._np_rng = np.random.default_rng()

        # 数据缓存
        self._X_train_np: Optional[np.ndarray] = None
        self._unique_vals_dict: Dict[int, np.ndarray] = {}
        self._feature_ranges: Optional[np.ndarray] = None

        # 警告控制（避免重复警告）
        self._categorical_fallback_warned: set = set()

    def update_data(self, X_train_np: np.ndarray) -> None:
        """更新训练数据（预计算分类值和特征范围）

        Args:
            X_train_np: (N, d) 训练数据
        """
        self._X_train_np = X_train_np
        self._precompute_categorical_values()
        self._compute_feature_ranges()

        # 【新增】自动计算 local_num（如果启用）
        if self.auto_compute_local_num:
            self._auto_compute_local_num()

    def _precompute_categorical_values(self) -> None:
        """预计算每个分类维的unique值"""
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

        # 仅在有失败时警告
        if failed_dims:
            warnings.warn(
                f"预计算分类值失败的维度: {failed_dims}，"
                f"这些维度将保持原值（无局部探索）"
            )

    def _compute_feature_ranges(self) -> None:
        """计算特征范围 (min, max)"""
        if self._X_train_np is None:
            return

        x = self._X_train_np
        mn = x.min(axis=0)
        mx = x.max(axis=0)
        self._feature_ranges = np.stack([mn, mx], axis=0)  # (2, d)

    def _compute_lcm(self, numbers: List[int]) -> int:
        """计算一组数字的最小公倍数 (LCM)

        Args:
            numbers: 整数列表

        Returns:
            最小公倍数
        """
        import math
        if not numbers:
            return self._local_num_manual

        lcm = numbers[0]
        for num in numbers[1:]:
            lcm = abs(lcm * num) // math.gcd(lcm, num)
        return lcm

    def _auto_compute_local_num(self) -> None:
        """自动计算 local_num（基于低水平离散变量的LCM）

        逻辑：
        1. 收集所有 ≤ exhaustive_level_threshold 的离散变量的水平数
        2. 计算这些水平数的LCM（最小公倍数）
        3. 如果LCM超过上限，使用上限值
        4. 如果没有低水平离散变量，保持手动设置值
        """
        if not self.variable_types or not self._unique_vals_dict:
            # 没有变量类型信息或unique值，保持手动设置
            self.local_num = self._local_num_manual
            return

        # 收集低水平离散变量的水平数
        low_level_counts = []

        for dim_idx, vtype in self.variable_types.items():
            if vtype in ("categorical", "integer"):
                # 获取该维度的水平数
                if dim_idx in self._unique_vals_dict:
                    n_levels = len(self._unique_vals_dict[dim_idx])
                else:
                    # 对于integer类型，从feature_ranges计算
                    if self._feature_ranges is not None and 0 <= dim_idx < self._feature_ranges.shape[1]:
                        min_val = self._feature_ranges[0, dim_idx]
                        max_val = self._feature_ranges[1, dim_idx]
                        n_levels = int(max_val - min_val) + 1
                    else:
                        continue

                # 只考虑低水平变量（≤ threshold）
                if n_levels <= self.exhaustive_level_threshold:
                    low_level_counts.append(n_levels)

        if not low_level_counts:
            # 没有低水平离散变量，保持手动设置
            self.local_num = self._local_num_manual
            warnings.warn(
                f"auto_compute_local_num=True 但没有找到低水平离散变量 "
                f"(≤{self.exhaustive_level_threshold}水平)，使用手动设置值 local_num={self._local_num_manual}",
                UserWarning
            )
            return

        # 计算LCM
        computed_lcm = self._compute_lcm(low_level_counts)

        # 应用上限
        if computed_lcm > self.auto_local_num_max:
            self.local_num = self.auto_local_num_max
            warnings.warn(
                f"自动计算的 local_num={computed_lcm} 超过上限 {self.auto_local_num_max}，"
                f"已限制为 {self.auto_local_num_max}。"
                f"低水平变量水平数: {low_level_counts}",
                UserWarning
            )
        else:
            self.local_num = computed_lcm
            print(
                f"[LocalSampler] 自动计算 local_num = {self.local_num} "
                f"(基于低水平变量: {low_level_counts}, LCM={computed_lcm})"
            )

    def sample(
        self,
        X_can_t: torch.Tensor,
        dims: Sequence[int]
    ) -> torch.Tensor:
        """生成局部扰动点

        Args:
            X_can_t: (B, d) 候选点
            dims: 要扰动的维度列表

        Returns:
            (B * local_num, d) 扰动后的点
        """
        # 【修复】移除全局 seed 调用，使用实例级 RNG
        B, d = X_can_t.shape

        # 获取特征范围
        if self._feature_ranges is None:
            mn = torch.zeros(d, dtype=X_can_t.dtype, device=X_can_t.device)
            mx = torch.ones(d, dtype=X_can_t.dtype, device=X_can_t.device)
        else:
            mn = torch.as_tensor(
                self._feature_ranges[0],
                dtype=X_can_t.dtype,
                device=X_can_t.device
            )
            mx = torch.as_tensor(
                self._feature_ranges[1],
                dtype=X_can_t.dtype,
                device=X_can_t.device
            )

        span = torch.clamp(mx - mn, min=1e-6)

        # 构造 (B, local_num, d)
        base = X_can_t.unsqueeze(1).repeat(1, self.local_num, 1)

        # 按维类型处理
        for k in dims:
            vt = self.variable_types.get(k) if self.variable_types else None

            if vt == "categorical":
                base = self._perturb_categorical(base, k, B)

            elif vt == "integer":
                base = self._perturb_integer(base, k, B, mn[k], mx[k], span[k])

            else:  # continuous or None
                base = self._perturb_continuous(base, k, B, mn[k], mx[k], span[k])

        return base.reshape(B * self.local_num, d)

    def _perturb_categorical(
        self,
        base: torch.Tensor,
        k: int,
        B: int
    ) -> torch.Tensor:
        """分类变量扰动：混合策略（穷举 vs 随机采样）

        策略选择：
        - 启用混合扰动 且 水平数 ≤ threshold → 穷举所有水平（完全覆盖）
        - 其他情况 → 随机采样（原始逻辑）
        """
        unique_vals = self._unique_vals_dict.get(k)

        if unique_vals is None or len(unique_vals) == 0:
            # 降级：保持原值
            if k not in self._categorical_fallback_warned:
                warnings.warn(
                    f"分类维 {k} 的unique值未找到，保持原值（该维度无探索贡献）"
                )
                self._categorical_fallback_warned.add(k)
            return base

        n_levels = len(unique_vals)

        # 【混合策略】判断是否使用穷举
        if (self.use_hybrid_perturbation and
            n_levels <= self.exhaustive_level_threshold):
            # ========== 穷举模式 ==========
            if self.exhaustive_use_cyclic_fill:
                # 循环填充到local_num（均衡覆盖所有水平）
                # 例如：3水平 + local_num=6 → [0,1,2,0,1,2]
                n_repeats = (self.local_num // n_levels) + 1
                samples = np.tile(unique_vals, (B, n_repeats))
                samples = samples[:, :self.local_num]  # 裁剪到local_num
            else:
                # 只生成n_levels个样本（不填充）
                # 例如：3水平 → [0,1,2] （忽略local_num）
                samples = np.tile(unique_vals, (B, 1))

            base[:, :samples.shape[1], k] = torch.from_numpy(samples).to(
                dtype=base.dtype, device=base.device
            )
        else:
            # ========== 随机采样模式（原始逻辑）==========
            samples = self._np_rng.choice(unique_vals, size=(B, self.local_num))
            base[:, :, k] = torch.from_numpy(samples).to(
                dtype=base.dtype, device=base.device
            )

        return base

    def _perturb_integer(
        self,
        base: torch.Tensor,
        k: int,
        B: int,
        mn: float,
        mx: float,
        span: float
    ) -> torch.Tensor:
        """整数变量扰动：混合策略（穷举 vs 高斯）

        策略选择：
        - 启用混合扰动 且 整数水平数 ≤ threshold → 穷举所有整数值
        - 其他情况 → 高斯扰动+舍入（原始逻辑）
        """
        # 计算整数范围内的所有可能值
        int_min = int(np.floor(mn.item() if torch.is_tensor(mn) else mn))
        int_max = int(np.ceil(mx.item() if torch.is_tensor(mx) else mx))
        all_integers = np.arange(int_min, int_max + 1)
        n_levels = len(all_integers)

        # 【混合策略】判断是否使用穷举
        if (self.use_hybrid_perturbation and
            n_levels <= self.exhaustive_level_threshold):
            # ========== 穷举模式 ==========
            if self.exhaustive_use_cyclic_fill:
                # 循环填充到local_num
                n_repeats = (self.local_num // n_levels) + 1
                samples = np.tile(all_integers, (B, n_repeats))
                samples = samples[:, :self.local_num]
            else:
                # 只生成n_levels个样本
                samples = np.tile(all_integers, (B, 1))

            base[:, :samples.shape[1], k] = torch.from_numpy(samples).to(
                dtype=base.dtype, device=base.device
            )
        else:
            # ========== 高斯扰动模式（原始逻辑）==========
            sigma = self.local_jitter_frac * span
            noise = torch.randn(B, self.local_num, device=base.device) * sigma
            base[:, :, k] = torch.round(
                torch.clamp(base[:, :, k] + noise, min=mn, max=mx)
            )

        return base

    def _perturb_continuous(
        self,
        base: torch.Tensor,
        k: int,
        B: int,
        mn: float,
        mx: float,
        span: float
    ) -> torch.Tensor:
        """连续变量扰动：高斯扰动"""
        sigma = self.local_jitter_frac * span
        noise = torch.randn(B, self.local_num, device=base.device) * sigma
        base[:, :, k] = torch.clamp(base[:, :, k] + noise, min=mn, max=mx)
        return base
