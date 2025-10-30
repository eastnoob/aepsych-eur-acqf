"""
Simple test to verify the module works.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")
print(f"Current directory: {Path.cwd()}")

try:
    from acquisition_function import VarianceReductionWithCoverageAcqf

    print("✓ Successfully imported VarianceReductionWithCoverageAcqf")

    # Test basic functionality
    np.random.seed(42)
    X = np.random.rand(20, 3)
    y = np.random.rand(20)

    acq_fn = VarianceReductionWithCoverageAcqf()
    print("✓ Successfully created acquisition function")

    acq_fn.fit(X, y)
    print("✓ Successfully fitted on data")

    X_candidates = np.random.rand(50, 3)
    scores = acq_fn(X_candidates)
    print(f"✓ Successfully evaluated {len(scores)} candidates")
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    next_X, next_idx = acq_fn.select_next(X_candidates, n_select=3)
    print(f"✓ Successfully selected {len(next_X)} points")

    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)

except Exception as e:
    print(f"\n✗ Error: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
