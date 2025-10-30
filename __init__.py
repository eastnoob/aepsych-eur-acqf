"""
Extensions: EUR family acquisition functions and utilities for AEPsych.

This package may include multiple variants (V1/V2/V3/V4). Some modules are optional
and guarded with try/except to avoid hard import failures if a file is missing.
"""

__version__ = "2.0.0"

# Optional: V1 (may be absent in some branches)
try:  # pragma: no cover - optional module
    from .acquisition_function import VarianceReductionWithCoverageAcqf
except Exception:  # pragma: no cover - fallback if file/symbol missing
    VarianceReductionWithCoverageAcqf = None  # type: ignore

# Optional: V2/V3
try:
    from .acquisition_function_v2 import EnhancedVarianceReductionAcqf
except Exception:  # pragma: no cover
    EnhancedVarianceReductionAcqf = None  # type: ignore

try:
    from .acquisition_function_v3 import HardExclusionAcqf, CombinedAcqf
except Exception:  # pragma: no cover
    HardExclusionAcqf = None  # type: ignore
    CombinedAcqf = None  # type: ignore

try:
    from .hard_exclusion_generator import HardExclusionGenerator
except Exception:  # pragma: no cover
    HardExclusionGenerator = None  # type: ignore

# V4 and core utilities
from .acquisition_function_v4 import EURAcqfV4
from .gower_distance import gower_distance, compute_coverage
from .gp_variance import GPVarianceCalculator

# Build public symbols list based on available imports
__all__ = [
    name
    for name, sym in {
        "VarianceReductionWithCoverageAcqf": VarianceReductionWithCoverageAcqf,
        "EnhancedVarianceReductionAcqf": EnhancedVarianceReductionAcqf,
        "HardExclusionAcqf": HardExclusionAcqf,
        "CombinedAcqf": CombinedAcqf,
        "HardExclusionGenerator": HardExclusionGenerator,
        "EURAcqfV4": EURAcqfV4,
        "gower_distance": gower_distance,
        "compute_coverage": compute_coverage,
        "GPVarianceCalculator": GPVarianceCalculator,
    }.items()
    if sym is not None
]
