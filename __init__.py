"""
Dynamic EUR Acquisition Function (V4) for AEPsych.

This package exposes the V4 EUR acquisition function and core utilities at the
root level. Legacy variants (V1/V2/V3) and auxiliary generators are available
under the `legacy/` subpackage to keep the root minimal while preserving
optional compatibility.
"""

__version__ = "2.1.0"
__author__ = "Fengxu Tian"

# Core V4 exports
from .acquisition_function_v4 import EURAcqfV4
from .gower_distance import gower_distance, compute_coverage_batch, compute_coverage
from .gp_variance import GPVarianceCalculator

# Optional legacy exports (best-effort; absent if legacy/ not present)
try:  # pragma: no cover
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
    "EURAcqfV4",
    "gower_distance",
    "compute_coverage_batch",
    "compute_coverage",
    "GPVarianceCalculator",
]

# Conditionally extend __all__ with legacy symbols if available
if VarianceReductionWithCoverageAcqf is not None:
    __all__.append("VarianceReductionWithCoverageAcqf")
if EnhancedVarianceReductionAcqf is not None:
    __all__.append("EnhancedVarianceReductionAcqf")
if HardExclusionAcqf is not None:
    __all__.append("HardExclusionAcqf")
if CombinedAcqf is not None:
    __all__.append("CombinedAcqf")
if HardExclusionGenerator is not None:
    __all__.append("HardExclusionGenerator")
