#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom Ordinal Transform - 有序参数支持

实现稀疏采样连续物理值的参数类型，如天花板高度[2.0, 2.5, 3.5]。
保留序关系和间距信息，使ANOVA能正确分解参数效应。

设计原则：
- Transform输出规范化值 [0.0, 0.333, 1.0]（保留间距比例）
- LocalSampler在规范化值空间扰动
- ANOVA看到正确的相对间距关系
"""

import warnings
from typing import Dict, List, Optional

import numpy as np
import torch


class CustomOrdinal:
    """
    有序参数Transform - 输出保留间距的规范化值

    支持两种类型：
    - custom_ordinal: 等差有序参数（自动计算）
    - custom_ordinal_mono: 非等差单调参数（手动指定）

    设计：
    - 物理值 [2.0, 2.5, 3.5] → 规范化值 [0.0, 0.333, 1.0]
    - 保留间距信息供GP和ANOVA使用
    - LocalSampler在规范化值空间扰动
    """

    def __init__(
        self,
        indices: List[int],
        values: Dict[int, List[float]],
        level_names: Optional[Dict[int, List[str]]] = None,
    ):
        """
        初始化Ordinal Transform

        Args:
            indices: 参数维度列表
            values: 各维度的物理值列表 {index: [2.0, 2.5, 3.5]}
            level_names: 可选字符串标签映射 {index: ["low", "medium", "high"]}
        """
        self.indices = indices
        self.values = values
        self.level_names = level_names or {}

        # 验证
        for idx in self.indices:
            if idx not in values:
                raise ValueError(f"Index {idx} not found in values dict")

            vals = values[idx]
            if len(vals) < 2:
                raise ValueError(
                    f"Index {idx}: must have at least 2 values, got {len(vals)}"
                )

            # 确保值是排序的
            sorted_vals = sorted(vals)
            if vals != sorted_vals:
                warnings.warn(
                    f"Index {idx}: values {vals} not sorted, will use {sorted_vals}"
                )
                self.values[idx] = sorted_vals

        # 构建规范化映射
        self._build_normalized_mappings()

    def _build_normalized_mappings(self):
        """
        构建物理值 ↔ 规范化值的双向映射

        规范化方法：Min-Max归一化到[0, 1]
        - 保留间距比例信息
        - ANOVA能看到正确的相对间距
        """
        self.normalized_values = {}  # {index: [norm_v0, norm_v1, ...]}
        self.physical_to_normalized = {}  # {index: {phys_val: norm_val}}
        self.normalized_to_physical = {}  # {index: {norm_val: phys_val}}

        for idx in self.indices:
            phys_vals = np.array(self.values[idx], dtype=np.float64)

            # Min-max归一化到[0, 1]
            min_val = phys_vals.min()
            max_val = phys_vals.max()

            if max_val - min_val < 1e-10:
                # 所有值相同，归一化为0
                norm_vals = np.zeros_like(phys_vals)
            else:
                norm_vals = (phys_vals - min_val) / (max_val - min_val)

            # 保存规范化值
            self.normalized_values[idx] = norm_vals

            # 构建双向字典（处理浮点精度）
            self.physical_to_normalized[idx] = {
                round(float(p), 10): round(float(n), 10)
                for p, n in zip(phys_vals, norm_vals)
            }
            self.normalized_to_physical[idx] = {
                round(float(n), 10): round(float(p), 10)
                for n, p in zip(norm_vals, phys_vals)
            }

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        物理值 → 规范化值

        Args:
            X: shape (N, d) 物理值，如 [[2.5], [3.5], [2.0]]

        Returns:
            X_normalized: shape (N, d) 规范化值，如 [[0.333], [1.0], [0.0]]
        """
        X_normalized = X.clone()

        for i, idx in enumerate(self.indices):
            phys_vals = X[..., i].cpu().numpy()
            norm_vals = np.zeros_like(phys_vals, dtype=np.float64)

            # 查表转换
            phys_to_norm = self.physical_to_normalized[idx]

            for j, pv in enumerate(phys_vals.flat):
                pv_rounded = round(float(pv), 10)

                if pv_rounded not in phys_to_norm:
                    # 最近邻匹配（容错）
                    closest = min(
                        phys_to_norm.keys(),
                        key=lambda x: abs(x - pv_rounded)
                    )
                    norm_vals.flat[j] = phys_to_norm[closest]
                    warnings.warn(
                        f"Value {pv} not in ordinal values, using nearest {closest}"
                    )
                else:
                    norm_vals.flat[j] = phys_to_norm[pv_rounded]

            X_normalized[..., i] = torch.from_numpy(norm_vals).to(
                dtype=X.dtype, device=X.device
            )

        return X_normalized

    def untransform(self, X: torch.Tensor) -> torch.Tensor:
        """
        规范化值 → 物理值

        Args:
            X: shape (N, d) 规范化值，如 [[0.333], [1.0], [0.0]]

        Returns:
            X_physical: shape (N, d) 物理值，如 [[2.5], [3.5], [2.0]]
        """
        X_physical = X.clone()

        for i, idx in enumerate(self.indices):
            norm_vals = X[..., i].cpu().numpy()
            phys_vals = np.zeros_like(norm_vals, dtype=np.float64)

            # 查表转换
            norm_to_phys = self.normalized_to_physical[idx]

            for j, nv in enumerate(norm_vals.flat):
                nv_rounded = round(float(nv), 10)

                if nv_rounded not in norm_to_phys:
                    # 最近邻匹配
                    closest = min(
                        norm_to_phys.keys(),
                        key=lambda x: abs(x - nv_rounded)
                    )
                    phys_vals.flat[j] = norm_to_phys[closest]
                else:
                    phys_vals.flat[j] = norm_to_phys[nv_rounded]

            X_physical[..., i] = torch.from_numpy(phys_vals).to(
                dtype=X.dtype, device=X.device
            )

        return X_physical

    def transform_bounds(
        self,
        X: torch.Tensor,
        epsilon: float = 1e-6
    ) -> torch.Tensor:
        """
        物理边界 → 规范化边界

        Args:
            X: shape (2, d) 边界 [[lb], [ub]]，物理值
            epsilon: 边界扩展量

        Returns:
            X_bounds: shape (2, d) 规范化边界，约为 [[-ε], [1.0+ε]]
        """
        X_bounds = X.clone()

        for i, idx in enumerate(self.indices):
            # 规范化后的边界总是[0, 1]
            X_bounds[0, i] = -epsilon  # 下界稍微扩展
            X_bounds[1, i] = 1.0 + epsilon  # 上界稍微扩展

        return X_bounds

    @staticmethod
    def _compute_arithmetic_sequence(
        min_val: float,
        max_val: float,
        step: Optional[float] = None,
        num_levels: Optional[int] = None
    ) -> np.ndarray:
        """
        自动计算等差数列

        Args:
            min_val: 最小值
            max_val: 最大值
            step: 步长（与num_levels二选一）
            num_levels: 等级数量（与step二选一）

        Returns:
            values: 等差数列数组
        """
        if step is not None:
            # 使用linspace避免累积误差
            num_steps = int(round((max_val - min_val) / step)) + 1
            values = np.linspace(min_val, max_val, num_steps)
        elif num_levels is not None:
            values = np.linspace(min_val, max_val, int(num_levels))
        else:
            raise ValueError("Must specify either 'step' or 'num_levels'")

        return values

    @classmethod
    def get_config_options(
        cls,
        config,
        name: str,
        options: Optional[dict] = None
    ) -> dict:
        """
        从配置解析ordinal参数

        配置优先级：
        1. values: 直接指定值列表（用于非等差）
        2. min_value + max_value + step: 自动计算等差（推荐）
        3. min_value + max_value + num_levels: 精确等分
        4. levels: 字符串标签（Likert量表）

        Args:
            config: 配置对象
            name: 参数名称
            options: 额外选项

        Returns:
            config_dict: Transform初始化参数
        """
        options = options or {}

        # 优先级1: 直接指定values
        if "values" in options:
            values = options["values"]
            if isinstance(values, str):
                # 解析字符串列表
                import ast
                values = ast.literal_eval(values)
            return {"indices": [0], "values": {0: list(values)}}

        # 优先级2: min/max + step 或 num_levels
        if "min_value" in options and "max_value" in options:
            min_val = float(options["min_value"])
            max_val = float(options["max_value"])

            if "step" in options:
                step = float(options["step"])
                values = cls._compute_arithmetic_sequence(
                    min_val, max_val, step=step
                )
            elif "num_levels" in options:
                num_levels = int(options["num_levels"])
                values = cls._compute_arithmetic_sequence(
                    min_val, max_val, num_levels=num_levels
                )
            else:
                raise ValueError(
                    f"[{name}] Must specify 'step' or 'num_levels' with min/max_value"
                )

            return {"indices": [0], "values": {0: list(values)}}

        # 优先级3: levels（字符串标签）
        if "levels" in options:
            levels = options["levels"]
            if isinstance(levels, str):
                levels = [s.strip() for s in levels.split(',')]

            # 字符串标签 → 整数序列（等差）
            values = list(range(len(levels)))
            return {
                "indices": [0],
                "values": {0: values},
                "level_names": {0: levels}
            }

        raise ValueError(
            f"[{name}] Must specify one of:\n"
            "  1. 'values' (direct list)\n"
            "  2. 'min_value' + 'max_value' + ('step' or 'num_levels')\n"
            "  3. 'levels' (string labels)"
        )
