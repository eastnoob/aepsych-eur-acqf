"""
覆盖度计算模块

基于Gower距离的空间填充性度量。
确保采样点远离已观测区域，提高探索性。

Example:
    >>> helper = CoverageHelper(X_train_np, variable_types)
    >>> coverage = helper.compute_coverage(X_candidates_torch)
"""

from __future__ import annotations
from typing import Optional, Dict
import numpy as np
import torch

try:  # pragma: no cover
    from ..gower_distance import compute_coverage_batch
except Exception:  # pragma: no cover
    from gower_distance import compute_coverage_batch  # type: ignore


class CoverageHelper:
    """覆盖度计算辅助类

    封装Gower距离计算逻辑，提供torch/numpy接口转换。
    """

    def __init__(
        self,
        X_train_np: Optional[np.ndarray] = None,
        variable_types: Optional[Dict[int, str]] = None,
        coverage_method: str = "min_distance"
    ):
        """
        Args:
            X_train_np: (N, d) 训练数据（numpy）
            variable_types: 变量类型字典
            coverage_method: 覆盖度计算方法
                           'min_distance': 到最近历史点的距离
                           'mean_distance': 到所有历史点的平均距离
        """
        self._X_train_np = X_train_np
        self.variable_types = variable_types
        self.coverage_method = coverage_method

    def update_training_data(self, X_train_np: np.ndarray) -> None:
        """更新训练数据

        Args:
            X_train_np: (N, d) 训练数据
        """
        self._X_train_np = X_train_np

    def compute_coverage(self, X_can_t: torch.Tensor) -> torch.Tensor:
        """计算覆盖度（torch接口）

        Args:
            X_can_t: (B, d) 候选点

        Returns:
            (B,) 覆盖度值（越大越好）
        """
        if self._X_train_np is None or self._X_train_np.shape[0] == 0:
            return torch.zeros(
                X_can_t.shape[0],
                dtype=X_can_t.dtype,
                device=X_can_t.device
            )

        # 转换为numpy
        X_np = X_can_t.detach().cpu().numpy()

        # 维度检查
        try:
            d_can = X_np.shape[1]
            d_hist = self._X_train_np.shape[1]
        except Exception:
            d_can = d_hist = -1

        if d_can != d_hist and d_can != -1 and d_hist != -1:
            return torch.zeros(
                X_np.shape[0],
                dtype=X_can_t.dtype,
                device=X_can_t.device
            )

        # 处理变量类型字典（只保留有效维度）
        vt = None
        if self.variable_types is not None and d_can >= 0:
            vt = {
                k: ("categorical" if v == "categorical" else "continuous")
                for k, v in self.variable_types.items()
                if 0 <= k < d_can
            }

        # 计算覆盖度
        try:
            cov_np = compute_coverage_batch(
                X_np,
                self._X_train_np,
                variable_types=vt,
                ranges=None,
                method=self.coverage_method
            )
            cov_t = torch.from_numpy(cov_np).to(
                dtype=X_can_t.dtype,
                device=X_can_t.device
            )
            return cov_t
        except Exception:
            return torch.zeros(
                X_np.shape[0],
                dtype=X_can_t.dtype,
                device=X_can_t.device
            )
