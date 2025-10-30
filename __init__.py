"""
Variance Reduction with Coverage Acquisition Function for AEPsych

This module implements an active learning acquisition function that combines
parameter variance reduction (information gain) with spatial coverage using
Gower distance for mixed-type variables.
"""

from .acquisition_function import VarianceReductionWithCoverageAcqf
from .acquisition_function_v2 import EnhancedVarianceReductionAcqf
from .acquisition_function_v3 import HardExclusionAcqf, CombinedAcqf
from .hard_exclusion_generator import HardExclusionGenerator
from .acquisition_function_v4 import EURAcqfV4
from .gower_distance import gower_distance, compute_coverage
from .gp_variance import GPVarianceCalculator

__all__ = [
    "VarianceReductionWithCoverageAcqf",
    "EnhancedVarianceReductionAcqf",
    "HardExclusionAcqf",
    "CombinedAcqf",
    "HardExclusionGenerator",
    "EURAcqfV4",
    "gower_distance",
    "compute_coverage",
    "GPVarianceCalculator",
]

__version__ = "2.0.0"
