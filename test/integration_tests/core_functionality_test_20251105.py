"""
Core functionality test for EURAnovaPairAcqf
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

print("=" * 70)
print("EURAnovaPairAcqf Core Functionality Test")
print("=" * 70)

# Test 1: Import test
print("\n[Test 1] Testing imports...")
try:
    from eur_anova_pair import EURAnovaPairAcqf
    from gower_distance import compute_coverage_batch
    from gp_variance import GPVarianceCalculator
    print("[OK] All core modules imported successfully")
except Exception as e:
    print(f"[FAIL] Import failed: {e}")
    sys.exit(1)

# Test 2: Create mock GP model
print("\n[Test 2] Creating mock GP model...")
try:
    from botorch.models import SingleTaskGP
    from gpytorch.likelihoods import GaussianLikelihood

    # Create training data
    X_train = torch.rand(10, 3)
    y_train = torch.randn(10, 1)  # Need 2D for SingleTaskGP

    # Create model
    model = SingleTaskGP(X_train, y_train, likelihood=GaussianLikelihood())
    model.eval()
    print("[OK] Mock model created successfully")
except Exception as e:
    print(f"[FAIL] Model creation failed: {e}")
    sys.exit(1)

# Test 3: Initialize acquisition function
print("\n[Test 3] Initializing acquisition function...")
try:
    acqf = EURAnovaPairAcqf(
        model=model,
        interaction_pairs=[(0, 1), (1, 2)],
        gamma=0.3,
        total_budget=30,
        random_seed=42
    )
    print("[OK] Acquisition function initialized successfully")
    print(f"  - Number of interaction pairs: {len(acqf._pairs)}")
    print(f"  - Pairs: {acqf._pairs}")
except Exception as e:
    print(f"[FAIL] Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test forward pass
print("\n[Test 4] Testing forward pass...")
try:
    X_candidates = torch.rand(5, 1, 3)  # Shape: (batch, q=1, d)
    scores = acqf(X_candidates)
    print("[OK] Forward computation successful")
    print(f"  - Number of candidates: {X_candidates.shape[0]}")
    print(f"  - Score shape: {scores.shape}")
    print(f"  - Score range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
except Exception as e:
    print(f"[FAIL] Forward computation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test with debug mode
print("\n[Test 5] Testing debug mode...")
try:
    acqf_debug = EURAnovaPairAcqf(
        model=model,
        interaction_pairs=[(0, 1), (1, 2)],
        debug_components=True,
        random_seed=42
    )
    scores_debug = acqf_debug(X_candidates)
    diagnostics = acqf_debug.get_diagnostics()

    print("[OK] Debug mode test successful")
    print(f"  - lambda_t: {diagnostics['lambda_t']:.4f}")
    print(f"  - gamma_t: {diagnostics['gamma_t']:.4f}")
    print(f"  - n_train: {diagnostics['n_train']}")
except Exception as e:
    print(f"[FAIL] Debug mode failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test with variable types
print("\n[Test 6] Testing mixed variable types...")
try:
    acqf_mixed = EURAnovaPairAcqf(
        model=model,
        interaction_pairs=[(0, 1)],
        variable_types={0: "categorical", 1: "integer", 2: "continuous"},
        random_seed=42
    )
    scores_mixed = acqf_mixed(X_candidates)
    print("[OK] Mixed variable types test successful")
    print(f"  - Variable types config: {acqf_mixed.variable_types}")
except Exception as e:
    print(f"[FAIL] Mixed variable types test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test gower distance
print("\n[Test 7] Testing Gower distance calculation...")
try:
    X_candidates_np = X_candidates.squeeze(1).numpy()  # Remove q dimension
    X_sampled_np = X_train.numpy()

    coverage = compute_coverage_batch(
        X_candidates_np,
        X_sampled_np,
        method="min_distance"
    )
    print("[OK] Gower distance calculation successful")
    print(f"  - Coverage shape: {coverage.shape}")
    print(f"  - Coverage range: [{coverage.min():.4f}, {coverage.max():.4f}]")
except Exception as e:
    print(f"[FAIL] Gower distance calculation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Test GP variance calculator
print("\n[Test 8] Testing GP variance calculator...")
try:
    gp_calc = GPVarianceCalculator()
    gp_calc.fit(X_train.numpy(), y_train.squeeze().numpy(), interaction_indices=[(0, 1)])
    y_pred, y_std = gp_calc.predict(X_candidates.squeeze(1).numpy(), return_std=True)

    print("[OK] GP variance calculation successful")
    print(f"  - Prediction shape: {y_pred.shape}")
    print(f"  - Std dev shape: {y_std.shape}")
except Exception as e:
    print(f"[FAIL] GP variance calculation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("[SUCCESS] All tests passed! Core functionality is working!")
print("=" * 70)
