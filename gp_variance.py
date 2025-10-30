"""
Gaussian Process variance calculation for linear models.

This module implements variance estimation for main effects and interaction effects
using linear Gaussian Process models (or Bayesian linear regression approximations).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy.linalg import cho_solve, cho_factor


class GPVarianceCalculator:
    """
    Calculate GP posterior variance for linear models with main and interaction effects.

    This class uses Bayesian linear regression to estimate variances for:
    1. Main effects: θ_j for each feature j
    2. Interaction effects: θ_jk for feature pairs (j, k)

    The model is: y = Φ(X) @ θ + ε, where ε ~ N(0, σ²I)

    The posterior is: θ | D ~ N(μ_post, Σ_post)
    where:
        Σ_post = (Φ^T Φ / σ² + Σ_prior^{-1})^{-1}
        μ_post = Σ_post @ (Φ^T y / σ²)

    Parameters
    ----------
    noise_variance : float, default=1.0
        Observation noise variance (σ²)
    prior_variance : float, default=1.0
        Prior variance for parameters (diagonal of Σ_prior)
    include_intercept : bool, default=True
        Whether to include an intercept term
    """

    def __init__(
        self,
        noise_variance: float = 1.0,
        prior_variance: float = 1.0,
        include_intercept: bool = True,
    ):
        self.noise_variance = noise_variance
        self.prior_variance = prior_variance
        self.include_intercept = include_intercept

        # Cached values for efficiency
        self._Sigma_post = None
        self._mu_post = None
        self._Phi = None
        self._y = None
        self._n_main_effects = None
        self._n_interaction_effects = None
        self._interaction_indices = None

    def _build_design_matrix(
        self, X: np.ndarray, interaction_indices: Optional[List[Tuple[int, int]]] = None
    ) -> np.ndarray:
        """
        Build design matrix Φ(X) with main effects and interaction effects.

        Parameters
        ----------
        X : np.ndarray
            Input data (n_samples, n_features)
        interaction_indices : List[Tuple[int, int]], optional
            List of feature pairs for interaction terms.
            E.g., [(0, 1), (1, 2)] for interactions between features (0,1) and (1,2)

        Returns
        -------
        np.ndarray
            Design matrix (n_samples, n_params)
        """
        # Convert to numpy if needed
        import torch

        if isinstance(X, torch.Tensor):
            X = X.cpu().detach().numpy()

        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        elif X.ndim > 2:
            # Flatten extra dimensions
            X = X.reshape(-1, X.shape[-1])

        n_samples, n_features = X.shape

        # Main effects
        Phi = X.copy()

        # Interaction effects
        if interaction_indices is not None and len(interaction_indices) > 0:
            interactions = []
            for j, k in interaction_indices:
                if j >= n_features or k >= n_features:
                    raise ValueError(
                        f"Interaction index ({j}, {k}) out of range for {n_features} features"
                    )
                interactions.append((X[:, j] * X[:, k]).reshape(-1, 1))

            if len(interactions) > 0:
                Phi = np.hstack([Phi] + interactions)

        # Add intercept
        if self.include_intercept:
            Phi = np.hstack([np.ones((n_samples, 1)), Phi])

        return Phi

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        interaction_indices: Optional[List[Tuple[int, int]]] = None,
    ) -> "GPVarianceCalculator":
        """
        Fit the model and compute posterior distribution.

        Parameters
        ----------
        X : np.ndarray
            Training inputs (n_samples, n_features)
        y : np.ndarray
            Training outputs (n_samples,)
        interaction_indices : List[Tuple[int, int]], optional
            List of feature pairs for interaction terms

        Returns
        -------
        self
        """
        # Build design matrix
        Phi = self._build_design_matrix(X, interaction_indices)

        n_samples, n_params = Phi.shape

        # Compute posterior covariance and mean
        # Σ_post = (Φ^T Φ / σ² + I / σ²_prior)^{-1}
        precision_matrix = (Phi.T @ Phi) / self.noise_variance + np.eye(
            n_params
        ) / self.prior_variance

        # Use Cholesky decomposition for numerical stability
        try:
            c, lower = cho_factor(precision_matrix)
            Sigma_post = cho_solve((c, lower), np.eye(n_params))
        except np.linalg.LinAlgError:
            # Fallback to regular inverse if Cholesky fails
            Sigma_post = np.linalg.inv(precision_matrix)

        # μ_post = Σ_post @ (Φ^T y / σ²)
        mu_post = Sigma_post @ (Phi.T @ y) / self.noise_variance

        # Cache values
        self._Sigma_post = Sigma_post
        self._mu_post = mu_post
        self._Phi = Phi
        self._X = X  # 保存原始 X 数据
        self._y = y
        self._n_main_effects = X.shape[1]
        self._interaction_indices = (
            interaction_indices if interaction_indices is not None else []
        )
        self._n_interaction_effects = len(self._interaction_indices)

        return self

    def predict(
        self, X: np.ndarray, return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict outputs for new inputs.

        Parameters
        ----------
        X : np.ndarray
            Test inputs (n_samples, n_features)
        return_std : bool, default=False
            Whether to return predictive standard deviation

        Returns
        -------
        y_pred : np.ndarray
            Predicted outputs (n_samples,)
        y_std : np.ndarray, optional
            Predictive standard deviations (n_samples,)
        """
        if self._mu_post is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Build design matrix for test data
        Phi_test = self._build_design_matrix(X, self._interaction_indices)

        # Mean prediction
        y_pred = Phi_test @ self._mu_post

        if return_std:
            # Predictive variance: σ²_pred = σ² + Φ* Σ_post Φ*^T
            var_pred = self.noise_variance + np.sum(
                (Phi_test @ self._Sigma_post) * Phi_test, axis=1
            )
            y_std = np.sqrt(var_pred)
            return y_pred, y_std

        return y_pred

    def get_parameter_variance(
        self, param_indices: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Get posterior variance for specified parameters.

        Parameters
        ----------
        param_indices : List[int], optional
            Indices of parameters to get variance for.
            If None, return variances for all parameters.

        Returns
        -------
        np.ndarray
            Posterior variances (diagonal of Σ_post)
        """
        if self._Sigma_post is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        var = np.diag(self._Sigma_post)

        if param_indices is not None:
            var = var[param_indices]

        return var

    def get_main_effect_variance(self, feature_idx: int) -> float:
        """
        Get posterior variance for a specific main effect.

        Parameters
        ----------
        feature_idx : int
            Index of the feature (0 to n_features-1)

        Returns
        -------
        float
            Posterior variance of θ_j
        """
        if self._Sigma_post is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        if feature_idx < 0 or feature_idx >= self._n_main_effects:
            raise ValueError(
                f"Feature index {feature_idx} out of range [0, {self._n_main_effects})"
            )

        # Account for intercept offset
        offset = 1 if self.include_intercept else 0
        param_idx = offset + feature_idx

        return self._Sigma_post[param_idx, param_idx]

    def get_interaction_effect_variance(self, interaction_idx: int) -> float:
        """
        Get posterior variance for a specific interaction effect.

        Parameters
        ----------
        interaction_idx : int
            Index in the interaction_indices list

        Returns
        -------
        float
            Posterior variance of θ_jk
        """
        if self._Sigma_post is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        if interaction_idx < 0 or interaction_idx >= self._n_interaction_effects:
            raise ValueError(
                f"Interaction index {interaction_idx} out of range [0, {self._n_interaction_effects})"
            )

        # Account for intercept and main effects offset
        offset = (1 if self.include_intercept else 0) + self._n_main_effects
        param_idx = offset + interaction_idx

        return self._Sigma_post[param_idx, param_idx]

    def compute_variance_reduction(
        self, X_new: np.ndarray, y_new: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute variance reduction for main and interaction effects if new data is added.

        This implements:
        ΔVar[θ_j] = Var[θ_j | D_t] - Var[θ_j | D_t ∪ {(x_new, y_new)}]

        Parameters
        ----------
        X_new : np.ndarray
            New candidate point(s) (n_candidates, n_features)
        y_new : np.ndarray, optional
            Expected output(s) for new point(s). If None, use current prediction.

        Returns
        -------
        main_var_reduction : np.ndarray
            Variance reduction for each main effect (n_main_effects,)
        interaction_var_reduction : np.ndarray
            Variance reduction for each interaction effect (n_interaction_effects,)
        """
        if self._Sigma_post is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Current variances
        var_current = np.diag(self._Sigma_post)

        # If y_new not provided, use prediction
        if y_new is None:
            y_new = self.predict(X_new)

        # Convert to numpy and ensure proper shape
        import torch

        if isinstance(X_new, torch.Tensor):
            X_new = X_new.cpu().detach().numpy()

        # Combine current and new data
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)
        elif X_new.ndim > 2:
            # Flatten batch dimensions
            X_new = X_new.reshape(-1, X_new.shape[-1])

        if isinstance(y_new, (int, float)):
            y_new = np.array([y_new])

        # 使用原始 X 数据而不是设计矩阵
        X_combined = np.vstack([self._X, X_new])
        y_combined = np.concatenate([self._y, y_new])

        # 为合并后的数据构建设计矩阵
        Phi_combined = self._build_design_matrix(X_combined, self._interaction_indices)

        # Refit with new data (without modifying current state)
        n_samples, n_params = Phi_combined.shape
        precision_matrix_new = (
            Phi_combined.T @ Phi_combined
        ) / self.noise_variance + np.eye(n_params) / self.prior_variance

        try:
            c, lower = cho_factor(precision_matrix_new)
            Sigma_post_new = cho_solve((c, lower), np.eye(n_params))
        except np.linalg.LinAlgError:
            Sigma_post_new = np.linalg.inv(precision_matrix_new)

        var_new = np.diag(Sigma_post_new)

        # Variance reduction
        var_reduction = var_current - var_new

        # Split into main and interaction effects
        offset = 1 if self.include_intercept else 0
        main_var_reduction = var_reduction[offset : offset + self._n_main_effects]

        if self._n_interaction_effects > 0:
            interaction_var_reduction = var_reduction[offset + self._n_main_effects :]
        else:
            interaction_var_reduction = np.array([])

        return main_var_reduction, interaction_var_reduction
