"""
Example usage of Dynamic EUR Acquisition Function.

This script demonstrates how to use the acquisition function for active learning.
"""

import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dynamic_eur_acquisition import DynamicEURAcquisitionFunction


def example_1_basic_usage():
    """Example 1: Basic usage with default parameters."""
    print("=" * 70)
    print("Example 1: Basic Usage with Default Parameters")
    print("=" * 70)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 30
    n_features = 3

    X_train = np.random.rand(n_samples, n_features)
    y_train = (
        X_train[:, 0]
        + 2 * X_train[:, 1]
        - X_train[:, 2]
        + 0.1 * np.random.randn(n_samples)
    )

    # Initialize acquisition function
    acq_fn = DynamicEURAcquisitionFunction()

    # Fit on training data
    acq_fn.fit(X_train, y_train)

    # Generate candidate points
    X_candidates = np.random.rand(100, n_features)

    # Evaluate acquisition scores
    scores = acq_fn(X_candidates)

    print(f"Training data shape: {X_train.shape}")
    print(f"Candidate points: {X_candidates.shape[0]}")
    print(f"Acquisition scores range: [{scores.min():.4f}, {scores.max():.4f}]")

    # Select next point to sample
    next_X, next_idx = acq_fn.select_next(X_candidates, n_select=1)
    print(f"Selected next point: {next_X[0]}")
    print(f"Score: {scores[next_idx[0]]:.4f}")

    # Check dynamic parameters
    print(f"\nCurrent λ_t: {acq_fn.get_current_lambda():.4f}")
    print(f"Variance reduction ratio r_t: {acq_fn.get_variance_reduction_ratio():.4f}")
    print()


def example_2_with_interactions():
    """Example 2: Using interaction terms."""
    print("=" * 70)
    print("Example 2: With Interaction Terms")
    print("=" * 70)

    # Generate data with interactions
    np.random.seed(42)
    n_samples = 40
    n_features = 4

    X_train = np.random.rand(n_samples, n_features)
    # Include interaction effects in the true function
    y_train = (
        X_train[:, 0]
        + X_train[:, 1]
        + 2 * X_train[:, 0] * X_train[:, 1]  # Interaction 0-1
        + 1.5 * X_train[:, 2] * X_train[:, 3]  # Interaction 2-3
        + 0.1 * np.random.randn(n_samples)
    )

    # Initialize with specific interactions
    acq_fn = DynamicEURAcquisitionFunction(
        interaction_terms=[(0, 1), (2, 3)], lambda_min=0.5, lambda_max=3.0
    )

    acq_fn.fit(X_train, y_train)

    # Evaluate candidates
    X_candidates = np.random.rand(100, n_features)
    scores, info_scores, cov_scores = acq_fn(X_candidates, return_components=True)

    print(f"Training data shape: {X_train.shape}")
    print(f"Interaction terms: {acq_fn.interaction_terms}")
    print(f"\nScore statistics:")
    print(f"  Total:    [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  Info:     [{info_scores.min():.4f}, {info_scores.max():.4f}]")
    print(f"  Coverage: [{cov_scores.min():.4f}, {cov_scores.max():.4f}]")

    print(f"\nCurrent λ_t: {acq_fn.get_current_lambda():.4f}")
    print()


def example_3_from_config():
    """Example 3: Load configuration from file."""
    print("=" * 70)
    print("Example 3: Load Configuration from File")
    print("=" * 70)

    # Path to config file
    config_path = Path(__file__).parent / "config_example.ini"

    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        print("Skipping this example.")
        return

    # Initialize from config
    acq_fn = DynamicEURAcquisitionFunction(config_ini_path=config_path)

    print(f"Loaded configuration from: {config_path}")
    print(f"Parameters:")
    print(f"  λ_min: {acq_fn.lambda_min}")
    print(f"  λ_max: {acq_fn.lambda_max}")
    print(f"  τ_1: {acq_fn.tau_1}")
    print(f"  τ_2: {acq_fn.tau_2}")
    print(f"  γ: {acq_fn.gamma}")
    print(f"  Interaction terms: {acq_fn.interaction_terms}")

    # Generate data
    np.random.seed(42)
    n_features = 4
    X_train = np.random.rand(30, n_features)
    y_train = np.random.rand(30)

    acq_fn.fit(X_train, y_train)

    X_candidates = np.random.rand(50, n_features)
    scores = acq_fn(X_candidates)

    print(f"\nEvaluated {len(X_candidates)} candidates")
    print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print()


def example_4_active_learning_loop():
    """Example 4: Active learning loop."""
    print("=" * 70)
    print("Example 4: Active Learning Loop")
    print("=" * 70)

    # True function (unknown to the learner)
    def true_function(x):
        return np.sin(3 * x[:, 0]) + np.cos(2 * x[:, 1]) + 0.5 * x[:, 0] * x[:, 1]

    # Initialize
    np.random.seed(42)
    n_features = 2
    n_initial = 10
    n_iterations = 15
    n_candidates = 200

    # Initial random samples
    X_train = np.random.rand(n_initial, n_features)
    y_train = true_function(X_train)

    # Create acquisition function
    acq_fn = DynamicEURAcquisitionFunction(gamma=0.4, lambda_min=0.2, lambda_max=2.0)

    print(f"Starting active learning with {n_initial} initial samples")
    print(f"Running {n_iterations} iterations\n")

    for iteration in range(n_iterations):
        # Fit model
        acq_fn.fit(X_train, y_train)

        # Generate candidates
        X_candidates = np.random.rand(n_candidates, n_features)

        # Select next point
        next_X, next_idx = acq_fn.select_next(X_candidates, n_select=1)
        next_y = true_function(next_X)

        # Add to training set
        X_train = np.vstack([X_train, next_X])
        y_train = np.concatenate([y_train, next_y])

        # Get current stats
        lambda_t = acq_fn.get_current_lambda()
        r_t = acq_fn.get_variance_reduction_ratio()

        if iteration % 5 == 0 or iteration == n_iterations - 1:
            print(
                f"Iteration {iteration + 1:2d}: "
                f"n_samples={len(X_train):3d}, "
                f"λ_t={lambda_t:.3f}, "
                f"r_t={r_t:.3f}"
            )

    print(f"\nFinal training set size: {len(X_train)}")
    print(f"Final λ_t: {acq_fn.get_current_lambda():.4f}")
    print(f"Final r_t: {acq_fn.get_variance_reduction_ratio():.4f}")
    print()


def example_5_mixed_variables():
    """Example 5: Mixed continuous and categorical variables."""
    print("=" * 70)
    print("Example 5: Mixed Continuous and Categorical Variables")
    print("=" * 70)

    np.random.seed(42)
    n_samples = 30
    n_features = 4

    # Mixed data: features 0,1 are continuous, features 2,3 are categorical
    X_train = np.random.rand(n_samples, n_features)
    X_train[:, 2] = np.random.randint(0, 3, n_samples)  # 3 categories
    X_train[:, 3] = np.random.randint(0, 2, n_samples)  # 2 categories

    y_train = (
        X_train[:, 0]
        + 2 * X_train[:, 1]
        + 0.5 * X_train[:, 2]
        + 0.1 * np.random.randn(n_samples)
    )

    # Specify variable types
    variable_types = {
        0: "continuous",
        1: "continuous",
        2: "categorical",
        3: "categorical",
    }

    # Initialize acquisition function
    acq_fn = DynamicEURAcquisitionFunction(variable_types=variable_types, gamma=0.5)

    acq_fn.fit(X_train, y_train, variable_types=variable_types)

    # Generate mixed candidates
    X_candidates = np.random.rand(100, n_features)
    X_candidates[:, 2] = np.random.randint(0, 3, 100)
    X_candidates[:, 3] = np.random.randint(0, 2, 100)

    scores = acq_fn(X_candidates)

    print(f"Training data shape: {X_train.shape}")
    print(f"Variable types: {variable_types}")
    print(f"Acquisition scores range: [{scores.min():.4f}, {scores.max():.4f}]")

    # Select best
    next_X, next_idx = acq_fn.select_next(X_candidates, n_select=3)
    print(f"\nTop 3 selected points:")
    for i, (x, idx) in enumerate(zip(next_X, next_idx)):
        print(f"  {i+1}. X={x}, score={scores[idx]:.4f}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Dynamic EUR Acquisition Function - Examples")
    print("=" * 70 + "\n")

    example_1_basic_usage()
    example_2_with_interactions()
    example_3_from_config()
    example_4_active_learning_loop()
    example_5_mixed_variables()

    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)
