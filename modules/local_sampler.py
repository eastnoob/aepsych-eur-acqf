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
        random_seed: Optional[int] = 42
    ):
        """
        Args:
            variable_types: 变量类型字典 {dim_idx: type_str}
                           type_str ∈ {'categorical', 'integer', 'continuous'}
            local_jitter_frac: 扰动幅度（相对特征范围）
            local_num: 每个候选点生成的扰动数
            random_seed: 随机种子（None表示不固定）
        """
        self.variable_types = variable_types
        self.local_jitter_frac = local_jitter_frac
        self.local_num = local_num
        self.random_seed = random_seed

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
        # 设置随机种子
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

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
        """分类变量扰动：从unique值离散采样"""
        unique_vals = self._unique_vals_dict.get(k)

        if unique_vals is None or len(unique_vals) == 0:
            # 降级：保持原值
            if k not in self._categorical_fallback_warned:
                warnings.warn(
                    f"分类维 {k} 的unique值未找到，保持原值（该维度无探索贡献）"
                )
                self._categorical_fallback_warned.add(k)
            return base
        else:
            # 离散采样（完全合法）
            samples = np.random.choice(unique_vals, size=(B, self.local_num))
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
        """整数变量扰动：高斯+舍入+夹值"""
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
