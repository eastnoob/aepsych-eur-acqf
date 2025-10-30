"""
Extensions: EUR family acquisition functions and utilities for AEPsych.

This package is streamlined to focus on V4 (EURAcqfV4) in the root. Legacy
variants (V1/V2/V3) and auxiliary generators are available under the
`legacy/` subpackage to avoid clutter while keeping optional compatibility.
"""

__version__ = "2.1.0"

# V4 and core utilities (kept in package root)
from .acquisition_function_v4 import EURAcqfV4
from .gower_distance import gower_distance, compute_coverage
from .gp_variance import GPVarianceCalculator

# Optional legacy variants are loaded from the `legacy` subpackage if present.
# These are not required for using V4.
try:  # pragma: no cover - optional legacy imports
    from .legacy.acquisition_function import VarianceReductionWithCoverageAcqf
except Exception:  # pragma: no cover
    VarianceReductionWithCoverageAcqf = None  # type: ignore

try:  # pragma: no cover
    from .legacy.acquisition_function_v2 import EnhancedVarianceReductionAcqf
except Exception:  # pragma: no cover
    EnhancedVarianceReductionAcqf = None  # type: ignore

try:  # pragma: no cover
    from .legacy.acquisition_function_v3 import HardExclusionAcqf, CombinedAcqf
except Exception:  # pragma: no cover
    HardExclusionAcqf = None  # type: ignore
    CombinedAcqf = None  # type: ignore

try:  # pragma: no cover
    from .legacy.hard_exclusion_generator import HardExclusionGenerator
except Exception:  # pragma: no cover
    HardExclusionGenerator = None  # type: ignore

__all__ = [
    name
    for name, sym in {
        # Core V4 API and utilities
        "EURAcqfV4": EURAcqfV4,
        "gower_distance": gower_distance,
        "compute_coverage": compute_coverage,
        "GPVarianceCalculator": GPVarianceCalculator,
        # Optional legacy symbols (available if legacy subpackage exists)
        "VarianceReductionWithCoverageAcqf": VarianceReductionWithCoverageAcqf,
        "EnhancedVarianceReductionAcqf": EnhancedVarianceReductionAcqf,
        "HardExclusionAcqf": HardExclusionAcqf,
        "CombinedAcqf": CombinedAcqf,
        "HardExclusionGenerator": HardExclusionGenerator,
    }.items()
    if sym is not None
]
