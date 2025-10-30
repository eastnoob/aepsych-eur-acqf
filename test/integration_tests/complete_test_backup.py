"""
Unit tests for Dynamic EUR Acquisition Function.
"""

import unittest
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dynamic_eur_acquisition import (
    DynamicEURAcquisitionFunction,
    gower_distance,
    compute_coverage,
    GPVarianceCalculator,
)


class TestGowerDistance(unittest.TestCase):
    """Test Gower distance calculations."""

    def test_continuous_only(self):
        """Test Gower distance with continuous variables only."""
        x1 = np.array([0.0, 0.0])
        x2 = np.array([1.0, 1.0])

        dist = gower_distance(x1, x2)

        # With default range [0,1], distance should be 1.0
        self.assertAlmostEqual(dist, 1.0)

    def test_identical_points(self):
        """Test distance between identical points."""
        x = np.array([0.5, 0.3, 0.8])

        dist = gower_distance(x, x)

        self.assertAlmostEqual(dist, 0.0)

    def test_categorical_variables(self):
        """Test Gower distance with categorical variables."""
        x1 = np.array([1.0, 2.0])
        x2 = np.array([1.0, 3.0])

        variable_types = {0: "categorical", 1: "categorical"}

        dist = gower_distance(x1, x2, variable_types)

        # First feature same (0), second different (1), average = 0.5
        self.assertAlmostEqual(dist, 0.5)

    def test_mixed_variables(self):
        """Test Gower distance with mixed variable types."""
        x1 = np.array([0.0, 1.0])
        x2 = np.array([1.0, 1.0])

        variable_types = {0: "continuous", 1: "categorical"}

        dist = gower_distance(x1, x2, variable_types)

        # Continuous: 1.0, categorical: 0.0, average = 0.5
        self.assertAlmostEqual(dist, 0.5)


class TestCoverage(unittest.TestCase):
    """Test spatial coverage calculations."""

    def test_empty_samples(self):
        """Test coverage with no existing samples."""
        x = np.array([0.5, 0.5])
        X_sampled = np.array([]).reshape(0, 2)

        cov = compute_coverage(x, X_sampled)

        # Should return maximum coverage
        self.assertEqual(cov, 1.0)

    def test_min_distance(self):
        """Test minimum distance coverage."""
        x = np.array([1.0, 1.0])
        X_sampled = np.array([[0.0, 0.0], [0.5, 0.5]])

        cov = compute_coverage(x, X_sampled, method="min_distance")

        # Minimum distance should be to [0.5, 0.5]
        self.assertGreater(cov, 0.0)
        self.assertLessEqual(cov, 1.0)


class TestGPVarianceCalculator(unittest.TestCase):
    """Test GP variance calculator."""

    def test_fit_predict(self):
        """Test basic fit and predict."""
        np.random.seed(42)
        X = np.random.rand(20, 3)
        y = X[:, 0] + 2 * X[:, 1] - X[:, 2]

        gp = GPVarianceCalculator()
        gp.fit(X, y)

        X_test = np.random.rand(5, 3)
        y_pred = gp.predict(X_test)

        self.assertEqual(len(y_pred), 5)

    def test_main_effect_variance(self):
        """Test main effect variance extraction."""
        np.random.seed(42)
        X = np.random.rand(30, 4)
        y = np.random.rand(30)

        gp = GPVarianceCalculator()
        gp.fit(X, y)

        for i in range(4):
            var = gp.get_main_effect_variance(i)
            self.assertGreater(var, 0.0)

    def test_interaction_effects(self):
        """Test with interaction terms."""
        np.random.seed(42)
        X = np.random.rand(30, 3)
        y = X[:, 0] + X[:, 1] + X[:, 0] * X[:, 1]

        interactions = [(0, 1)]
        gp = GPVarianceCalculator()
        gp.fit(X, y, interaction_indices=interactions)

        # Should have main effects and interaction
        var_inter = gp.get_interaction_effect_variance(0)
        self.assertGreater(var_inter, 0.0)

    def test_variance_reduction(self):
        """Test variance reduction computation."""
        np.random.seed(42)
        X = np.random.rand(20, 3)
        y = np.random.rand(20)

        gp = GPVarianceCalculator()
        gp.fit(X, y)

        X_new = np.array([[0.5, 0.5, 0.5]])
        main_var_red, inter_var_red = gp.compute_variance_reduction(X_new)

        # Should have variance reduction for main effects
        self.assertEqual(len(main_var_red), 3)
        # All reductions should be non-negative
        self.assertTrue(np.all(main_var_red >= -1e-6))  # Allow small numerical errors


class TestDynamicEURAcquisitionFunction(unittest.TestCase):
    """Test main acquisition function."""

    def test_initialization(self):
        """Test initialization with default parameters."""
        acq_fn = DynamicEURAcquisitionFunction()

        self.assertEqual(acq_fn.lambda_min, 0.2)
        self.assertEqual(acq_fn.lambda_max, 2.0)
        self.assertEqual(acq_fn.tau_1, 0.5)
        self.assertEqual(acq_fn.tau_2, 0.1)
        self.assertEqual(acq_fn.gamma, 0.3)

    def test_fit(self):
        """Test fitting on data."""
        np.random.seed(42)
        X = np.random.rand(30, 3)
        y = np.random.rand(30)

        acq_fn = DynamicEURAcquisitionFunction()
        acq_fn.fit(X, y)

        self.assertTrue(acq_fn._fitted)
        self.assertEqual(acq_fn._n_features, 3)

    def test_call(self):
        """Test evaluating acquisition function."""
        np.random.seed(42)
        X_train = np.random.rand(30, 3)
        y_train = np.random.rand(30)

        acq_fn = DynamicEURAcquisitionFunction()
        acq_fn.fit(X_train, y_train)

        X_candidates = np.random.rand(50, 3)
        scores = acq_fn(X_candidates)

        self.assertEqual(len(scores), 50)
        self.assertTrue(np.all(np.isfinite(scores)))

    def test_return_components(self):
        """Test returning score components."""
        np.random.seed(42)
        X_train = np.random.rand(30, 3)
        y_train = np.random.rand(30)

        acq_fn = DynamicEURAcquisitionFunction()
        acq_fn.fit(X_train, y_train)

        X_candidates = np.random.rand(50, 3)
        total, info, cov = acq_fn(X_candidates, return_components=True)

        self.assertEqual(len(total), 50)
        self.assertEqual(len(info), 50)
        self.assertEqual(len(cov), 50)

        # Total should be sum of components
        np.testing.assert_array_almost_equal(total, info + cov)

    def test_select_next(self):
        """Test selecting next points."""
        np.random.seed(42)
        X_train = np.random.rand(30, 3)
        y_train = np.random.rand(30)

        acq_fn = DynamicEURAcquisitionFunction()
        acq_fn.fit(X_train, y_train)

        X_candidates = np.random.rand(100, 3)
        next_X, next_idx = acq_fn.select_next(X_candidates, n_select=5)

        self.assertEqual(next_X.shape, (5, 3))
        self.assertEqual(len(next_idx), 5)

    def test_with_interactions(self):
        """Test with interaction terms."""
        np.random.seed(42)
        X_train = np.random.rand(30, 4)
        y_train = np.random.rand(30)

        acq_fn = DynamicEURAcquisitionFunction(interaction_terms=[(0, 1), (2, 3)])
        acq_fn.fit(X_train, y_train)

        X_candidates = np.random.rand(50, 4)
        scores = acq_fn(X_candidates)

        self.assertEqual(len(scores), 50)
        self.assertEqual(len(acq_fn.interaction_terms), 2)

    def test_dynamic_lambda(self):
        """Test dynamic lambda computation."""
        np.random.seed(42)
        X = np.random.rand(30, 3)
        y = np.random.rand(30)

        acq_fn = DynamicEURAcquisitionFunction()
        acq_fn.fit(X, y)

        lambda_t = acq_fn.get_current_lambda()

        self.assertGreaterEqual(lambda_t, acq_fn.lambda_min)
        self.assertLessEqual(lambda_t, acq_fn.lambda_max)

    def test_config_loading(self):
        """Test loading configuration from file."""
        # Create temporary config file
        config_content = """[AcquisitionFunction]
lambda_min = 0.5
lambda_max = 3.0
tau_1 = 0.6
tau_2 = 0.15
gamma = 0.5
interaction_terms = (0,1);(1,2)
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            acq_fn = DynamicEURAcquisitionFunction(config_ini_path=config_path)

            self.assertEqual(acq_fn.lambda_min, 0.5)
            self.assertEqual(acq_fn.lambda_max, 3.0)
            self.assertEqual(acq_fn.tau_1, 0.6)
            self.assertEqual(acq_fn.tau_2, 0.15)
            self.assertEqual(acq_fn.gamma, 0.5)
            self.assertEqual(acq_fn.interaction_terms, [(0, 1), (1, 2)])
        finally:
            os.unlink(config_path)


class TestIntegration(unittest.TestCase):
    """Integration tests."""

    def test_active_learning_loop(self):
        """Test a simple active learning loop."""
        np.random.seed(42)

        # True function
        def f(x):
            return np.sin(3 * x[:, 0]) + np.cos(2 * x[:, 1])

        # Initial data
        X_train = np.random.rand(10, 2)
        y_train = f(X_train)

        acq_fn = DynamicEURAcquisitionFunction(gamma=0.4)

        # Run a few iterations
        for _ in range(5):
            acq_fn.fit(X_train, y_train)

            X_candidates = np.random.rand(100, 2)
            next_X, _ = acq_fn.select_next(X_candidates, n_select=1)
            next_y = f(next_X)

            X_train = np.vstack([X_train, next_X])
            y_train = np.concatenate([y_train, next_y])

        # Should have added 5 samples
        self.assertEqual(len(X_train), 15)
        self.assertEqual(len(y_train), 15)


if __name__ == "__main__":
    unittest.main()
