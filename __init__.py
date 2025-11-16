"""
Dynamic EUR ANOVA Acquisition Functions for AEPsych.

This package provides ANOVA-based acquisition functions with support for
multi-order interactions, mixed-type variables, and dynamic weight adjustment.

Core Components:
- EURAnovaMultiAcqf: NEW! Multi-order interactions (main + 2nd + 3rd + ...)
- EURAnovaPairAcqf: Legacy pair-wise interactions (backward compatible)
- gower_distance: Distance calculation for mixed-type variables
- compute_coverage_batch: Spatial coverage computation
- GPVarianceCalculator: Gaussian Process variance estimation utilities

Modular Architecture (NEW in v4.0):
- modules.anova_effects: ANOVA effect engine (extensible to any order)
- modules.ordinal_metrics: Ordinal model entropy calculation
- modules.dynamic_weights: Adaptive weight system (λ_t, γ_t)
- modules.local_sampler: Mixed-type local perturbation
- modules.coverage: Coverage computation
- modules.config_parser: Configuration parsing utilities
- modules.diagnostics: Debugging and diagnostics tools
"""

__version__ = "4.0.0"
__author__ = "Fengxu Tian"

# Core exports
from .eur_anova_multi import EURAnovaMultiAcqf  # NEW! Recommended
from .eur_anova_pair import EURAnovaPairAcqf    # Legacy (backward compatible)
from .gower_distance import gower_distance, compute_coverage_batch, compute_coverage
from .gp_variance import GPVarianceCalculator

# Module exports (for advanced users)
from . import modules

__all__ = [
    # Main acquisition functions
    "EURAnovaMultiAcqf",     # NEW! Use this for new projects
    "EURAnovaPairAcqf",      # Legacy support
    # Distance & Coverage
    "gower_distance",
    "compute_coverage_batch",
    "compute_coverage",
    # Utilities
    "GPVarianceCalculator",
    # Modules
    "modules",
]
