"""
V3 Acquisition Function: V1 + Hard Exclusion (migrated to legacy)
"""

import sys
import os
import numpy as np
import torch
from configparser import ConfigParser

# Relative import for VarianceReductionWithCoverageAcqf from legacy (placeholder)
from .acquisition_function import VarianceReductionWithCoverageAcqf  # type: ignore


class HardExclusionAcqf(VarianceReductionWithCoverageAcqf):  # type: ignore
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._sampled_designs = set()
        self._hard_exclusion_penalty = -1e10

    def _design_to_key(self, x):
        if isinstance(x, np.ndarray):
            if x.ndim > 1:
                x = x.flatten()
            return "_".join([f"{float(val):.6f}" for val in x])
        return str(x)

    def forward(self, X: "torch.Tensor") -> "torch.Tensor":  # pragma: no cover
        return super().forward(X)

    def __call__(self, X_candidates, return_components=False):  # pragma: no cover
        return super().__call__(X_candidates, return_components=return_components)


class CombinedAcqf(VarianceReductionWithCoverageAcqf):  # type: ignore
    def __init__(self, candidate_unsampled_ratio=0.8, **kwargs):
        super().__init__(**kwargs)
        self._sampled_designs = set()
        self.candidate_unsampled_ratio = candidate_unsampled_ratio

    def __call__(self, X_candidates, return_components=False):  # pragma: no cover
        return super().__call__(X_candidates, return_components=return_components)

    def _design_to_key(self, x):
        if isinstance(x, np.ndarray):
            if x.ndim > 1:
                x = x.flatten()
            return "_".join([f"{float(val):.6f}" for val in x])
        return str(x)
