"""
Test SPS discrete-aware skeleton point generation

Verifies that SPS uses median instead of midpoint for discrete dimensions.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
from modules.dynamic_weights import SPS_Tracker


def test_sps_continuous_only():
    """Test SPS with continuous variables only (baseline behavior)"""
    bounds = torch.tensor([[0.0, 0.0], [10.0, 10.0]])

    tracker = SPS_Tracker(bounds=bounds, variable_types=None)

    # Generate skeleton points without training data
    skeleton = tracker._generate_skeleton_points()

    # Center should be midpoint
    assert skeleton.shape == (5, 2)  # 2*d + 1 = 5
    assert torch.allclose(skeleton[0], torch.tensor([5.0, 5.0]))

    print("[PASS] Continuous-only test passed")


def test_sps_discrete_with_midpoint():
    """Test SPS with discrete variables using midpoint (no training data)"""
    bounds = torch.tensor([[-1.0, 0.0], [1.0, 2.0]])
    variable_types = {0: 'categorical', 1: 'ordinal'}

    tracker = SPS_Tracker(bounds=bounds, variable_types=variable_types)

    # Generate skeleton points without training data (should use midpoint)
    skeleton = tracker._generate_skeleton_points(X_train=None)

    # Center should be midpoint (no training data available)
    assert skeleton.shape == (5, 2)
    assert torch.allclose(skeleton[0], torch.tensor([0.0, 1.0]))

    print("[PASS] Discrete with midpoint test passed")


def test_sps_discrete_with_median():
    """Test SPS with discrete variables using median (with training data)"""
    bounds = torch.tensor([[-1.0, 0.0], [1.0, 2.0]])
    variable_types = {0: 'categorical', 1: 'ordinal'}

    # Training data: dim 0 has values [-1, 1], dim 1 has values [0, 1, 2]
    X_train = torch.tensor([
        [-1.0, 0.0],
        [-1.0, 1.0],
        [1.0, 2.0],
        [1.0, 1.0],
    ])

    tracker = SPS_Tracker(bounds=bounds, variable_types=variable_types)

    # Generate skeleton points with training data (should use median)
    skeleton = tracker._generate_skeleton_points(X_train=X_train)

    # Center should use median:
    # - dim 0: unique values [-1, 1], median_idx = 2//2 = 1, so median = 1.0
    # - dim 1: unique values [0, 1, 2], median_idx = 3//2 = 1, so median = 1.0
    assert skeleton.shape == (5, 2)
    assert torch.allclose(skeleton[0], torch.tensor([1.0, 1.0]))

    print("[PASS] Discrete with median test passed")


def test_sps_mixed_variables():
    """Test SPS with mixed continuous and discrete variables"""
    bounds = torch.tensor([[0.0, -1.0, 0.0], [10.0, 1.0, 2.0]])
    variable_types = {1: 'categorical', 2: 'ordinal'}  # dim 0 is continuous

    X_train = torch.tensor([
        [2.5, -1.0, 0.0],
        [7.5, 1.0, 1.0],
        [5.0, -1.0, 2.0],
        [3.0, 1.0, 1.0],
    ])

    tracker = SPS_Tracker(bounds=bounds, variable_types=variable_types)
    skeleton = tracker._generate_skeleton_points(X_train=X_train)

    # Center should be:
    # - dim 0 (continuous): midpoint = 5.0
    # - dim 1 (categorical): unique=[-1, 1], median_idx=1, median=1.0
    # - dim 2 (ordinal): unique=[0, 1, 2], median_idx=1, median=1.0
    assert skeleton.shape == (7, 3)  # 2*3 + 1 = 7
    assert torch.allclose(skeleton[0], torch.tensor([5.0, 1.0, 1.0]))

    print("[PASS] Mixed variables test passed")


def test_sps_binary_variable_fix():
    """Test that binary variables use median instead of invalid midpoint"""
    bounds = torch.tensor([[-1.0], [1.0]])
    variable_types = {0: 'categorical'}

    X_train = torch.tensor([[-1.0], [1.0], [-1.0], [1.0]])

    tracker = SPS_Tracker(bounds=bounds, variable_types=variable_types)
    skeleton = tracker._generate_skeleton_points(X_train=X_train)

    # Center should be median (-1 or 1), NOT midpoint (0)
    center_value = skeleton[0, 0].item()
    assert center_value in [-1.0, 1.0], f"Expected -1 or 1, got {center_value}"
    assert center_value != 0.0, "Center should not be invalid midpoint 0"

    print(f"[PASS] Binary variable fix test passed (center={center_value})")


def test_sps_integration_with_model():
    """Test SPS integration with lazy skeleton generation"""
    from unittest.mock import Mock

    bounds = torch.tensor([[-1.0, 0.0], [1.0, 2.0]])
    variable_types = {0: 'categorical', 1: 'ordinal'}

    tracker = SPS_Tracker(bounds=bounds, variable_types=variable_types)

    # Initially, skeleton_points should be None (lazy generation)
    assert tracker.skeleton_points is None

    # Create mock model with training data
    mock_model = Mock()
    X_train = torch.tensor([[-1.0, 0.0], [1.0, 1.0], [-1.0, 2.0]])
    mock_model.train_inputs = (X_train,)
    mock_model.training = False

    # Mock posterior
    mock_posterior = Mock()
    mock_posterior.mean = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).unsqueeze(-1)
    mock_model.posterior.return_value = mock_posterior

    # First call should generate skeleton points
    r_t = tracker.compute_r_t(mock_model)

    # Skeleton points should now be generated
    assert tracker.skeleton_points is not None
    assert tracker.skeleton_points.shape == (5, 2)

    # Center should use median
    # unique=[-1, 1], median_idx=1, median=1.0
    assert torch.allclose(tracker.skeleton_points[0], torch.tensor([1.0, 1.0]))

    print("[PASS] Integration test passed")


if __name__ == "__main__":
    print("Running SPS discrete-aware tests...\n")

    test_sps_continuous_only()
    test_sps_discrete_with_midpoint()
    test_sps_discrete_with_median()
    test_sps_mixed_variables()
    test_sps_binary_variable_fix()
    test_sps_integration_with_model()

    print("\n[SUCCESS] All tests passed!")

