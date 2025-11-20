"""
Dynamic EUR ANOVA Pair Acquisition Function for AEPsych.

This package provides the EURAnovaPairAcqf acquisition function with
ANOVA decomposition for mixed-type variables and dynamic weight adjustment.

Core Components:
- EURAnovaPairAcqf: Main acquisition function with ANOVA pair-wise interactions
- gower_distance: Distance calculation for mixed-type variables
- compute_coverage_batch: Spatial coverage computation
- GPVarianceCalculator: Gaussian Process variance estimation utilities
"""

__version__ = "3.0.0"
__author__ = "Fengxu Tian"

# Core exports
from .eur_anova_pair import EURAnovaPairAcqf
from .gower_distance import gower_distance, compute_coverage_batch, compute_coverage
from .gp_variance import GPVarianceCalculator

__all__ = [
    "EURAnovaPairAcqf",
    "gower_distance",
    "compute_coverage_batch",
    "compute_coverage",
    "GPVarianceCalculator",
]
