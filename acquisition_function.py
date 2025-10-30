"""
Variance Reduction with Coverage Acquisition Function.

This module implements an acquisition function that combines parameter variance reduction
(information gain) with spatial coverage for active learning in experimental design.
"""

import numpy as np
import configparser
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

# BoTorch imports
from botorch.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform

# 支持相对导入和直接导入
try:
    from .gower_distance import compute_coverage_batch, compute_coverage
    from .gp_variance import GPVarianceCalculator
except ImportError:
    from gower_distance import compute_coverage_batch, compute_coverage
    from gp_variance import GPVarianceCalculator


class VarianceReductionWithCoverageAcqf(AcquisitionFunction):
    """
    Variance Reduction with Coverage Acquisition Function for Active Learning.

    This acquisition function combines:
    1. Parameter variance reduction (information gain) for main and interaction effects
    2. Spatial coverage using Gower distance for mixed-type variables
    3. Dynamic weighting that adapts based on variance reduction progress

    The acquisition score is:
    α(x; D_t) = α_info(x; D_t) + α_cov(x; D_t)

    where:
    - α_info: Expected variance reduction for parameters (main + λ_t * interaction effects)
    - α_cov: Spatial coverage score using Gower distance

    Parameters
    ----------
    config_ini_path : str or Path, optional
        Path to configuration ini file. If None, use default parameters.
    lambda_min : float, default=0.2
        Minimum weight for interaction effects
    lambda_max : float, default=2.0
        Maximum weight for interaction effects
    tau_1 : float, default=0.5
        Upper threshold for relative variance
    tau_2 : float, default=0.1
        Lower threshold for relative variance
    gamma : float, default=0.3
        Weight for spatial coverage term
    interaction_terms : List[Tuple[int, int]], optional
        List of interaction terms (feature index pairs).
        If None, only main effects are used.
    noise_variance : float, default=1.0
        GP noise variance
    prior_variance : float, default=1.0
        Prior variance for parameters
    variable_types : Dict[int, str], optional
        Variable types for Gower distance ('continuous' or 'categorical')
    coverage_method : str, default='min_distance'
        Method for computing coverage ('min_distance', 'mean_distance', 'median_distance')

    Examples
    --------
    >>> # Initialize with default parameters
    >>> acq_fn = VarianceReductionWithCoverageAcqf()
    >>>
    >>> # Initialize from config file
    >>> acq_fn = VarianceReductionWithCoverageAcqf('config.ini')
    >>>
    >>> # Fit on data
    >>> X = np.random.rand(20, 3)
    >>> y = np.random.rand(20)
    >>> acq_fn.fit(X, y)
    >>>
    >>> # Evaluate candidates
    >>> X_candidates = np.random.rand(100, 3)
    >>> scores = acq_fn(X_candidates)
    >>> best_idx = np.argmax(scores)
    """

    def __init__(
        self,
        model: Model,
        config_ini_path: Optional[Union[str, Path]] = None,
        lambda_min: float = 0.2,
        lambda_max: float = 2.0,
        tau_1: float = 0.5,
        tau_2: float = 0.1,
        gamma: float = 0.3,
        interaction_terms: Optional[List[Tuple[int, int]]] = None,
        noise_variance: float = 1.0,
        prior_variance: float = 1.0,
        variable_types: Optional[Dict[int, str]] = None,
        coverage_method: str = "min_distance",
    ):
        """Initialize acquisition function.

        Parameters
        ----------
        model : Model
            The BoTorch/GPyTorch model to use for predictions
        """
        super().__init__(model=model)

        # Load config if provided
        if config_ini_path is not None:
            config = self._load_config(config_ini_path)
            lambda_min = config.get("lambda_min", lambda_min)
            lambda_max = config.get("lambda_max", lambda_max)
            tau_1 = config.get("tau_1", tau_1)
            tau_2 = config.get("tau_2", tau_2)
            gamma = config.get("gamma", gamma)
            interaction_terms = config.get("interaction_terms", interaction_terms)
            noise_variance = config.get("noise_variance", noise_variance)
            prior_variance = config.get("prior_variance", prior_variance)
            coverage_method = config.get("coverage_method", coverage_method)

        # Store parameters
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.gamma = gamma

        # Parse interaction_terms if it's a string
        if isinstance(interaction_terms, str):
            self.interaction_terms = self._parse_interaction_terms(interaction_terms)
        else:
            self.interaction_terms = (
                interaction_terms if interaction_terms is not None else []
            )
        self.noise_variance = noise_variance
        self.prior_variance = prior_variance
        self.variable_types = variable_types
        self.coverage_method = coverage_method

        # Initialize GP calculator
        self.gp_calculator = GPVarianceCalculator(
            noise_variance=noise_variance,
            prior_variance=prior_variance,
            include_intercept=True,
        )

        # Cache for efficiency
        self._X_train = None
        self._y_train = None
        self._var_initial = None
        self._var_current = None
        self._n_features = None
        self._fitted = False

    def _parse_interaction_terms(self, terms_str: str) -> List[Tuple[int, int]]:
        """Parse interaction terms from string format.

        Parameters
        ----------
        terms_str : str
            String like "(0,1);(1,3);(2,3)"

        Returns
        -------
        List[Tuple[int, int]]
            List of interaction term pairs
        """
        if not terms_str or not terms_str.strip():
            return []

        terms = []
        for term in terms_str.split(";"):
            term = term.strip()
            if term:
                # Remove parentheses and split
                term = term.strip("()")
                parts = term.split(",")
                if len(parts) == 2:
                    try:
                        i, j = int(parts[0].strip()), int(parts[1].strip())
                        terms.append((i, j))
                    except ValueError:
                        # Skip invalid terms
                        pass
        return terms

    def _load_config(self, config_path: Union[str, Path]) -> Dict:
        """
        Load configuration from ini file.

        Supports two formats:
        1. AEPsych standard format: [VarianceReductionWithCoverageAcqf] section
        2. Legacy format: [AcquisitionFunction] section
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        parser = configparser.ConfigParser()
        parser.read(config_path, encoding="utf-8")

        config = {}

        # Try AEPsych standard format first
        section_name = None
        if "VarianceReductionWithCoverageAcqf" in parser:
            section_name = "VarianceReductionWithCoverageAcqf"
        elif "AcquisitionFunction" in parser:  # Legacy format
            section_name = "AcquisitionFunction"

        if section_name is not None:
            section = parser[section_name]

            # Helper function to parse values (strips comments)
            def parse_value(value_str):
                """Parse value from config, removing inline comments."""
                # Split on # to remove comments
                value_str = value_str.split("#")[0].strip()
                return value_str

            # Load numeric parameters
            if "lambda_min" in section:
                config["lambda_min"] = float(parse_value(section["lambda_min"]))
            if "lambda_max" in section:
                config["lambda_max"] = float(parse_value(section["lambda_max"]))
            if "tau_1" in section:
                config["tau_1"] = float(parse_value(section["tau_1"]))
            if "tau_2" in section:
                config["tau_2"] = float(parse_value(section["tau_2"]))
            if "gamma" in section:
                config["gamma"] = float(parse_value(section["gamma"]))
            if "noise_variance" in section:
                config["noise_variance"] = float(parse_value(section["noise_variance"]))
            if "prior_variance" in section:
                config["prior_variance"] = float(parse_value(section["prior_variance"]))
            if "coverage_method" in section:
                config["coverage_method"] = parse_value(section["coverage_method"])

            # Load interaction terms
            # Format: (A,B);(B,C) or (0,1);(1,2)
            if "interaction_terms" in section:
                terms_str = parse_value(section["interaction_terms"])
                if terms_str.strip():
                    terms = []
                    for term in terms_str.split(";"):
                        term = term.strip()
                        if term:
                            # Remove parentheses and split
                            term = term.strip("()")
                            parts = term.split(",")
                            if len(parts) == 2:
                                try:
                                    # Try to parse as integers (feature indices)
                                    i, j = int(parts[0].strip()), int(parts[1].strip())
                                    terms.append((i, j))
                                except ValueError:
                                    # If not integers, keep as strings (feature names)
                                    # Will need to be resolved later
                                    terms.append((parts[0].strip(), parts[1].strip()))
                    config["interaction_terms"] = terms

        return config

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_types: Optional[Dict[int, str]] = None,
    ) -> "VarianceReductionWithCoverageAcqf":
        """
        Fit the acquisition function on current data.

        Parameters
        ----------
        X : np.ndarray
            Training inputs (n_samples, n_features)
        y : np.ndarray
            Training outputs (n_samples,)
        variable_types : Dict[int, str], optional
            Variable types for Gower distance. Overrides initialization value.

        Returns
        -------
        self
        """
        if variable_types is not None:
            self.variable_types = variable_types

        self._X_train = X.copy()
        self._y_train = y.copy()
        self._n_features = X.shape[1]

        # Fit GP model
        self.gp_calculator.fit(X, y, self.interaction_terms)

        # Store initial variances (from first fit, or use current if not available)
        if self._var_initial is None:
            # First time fitting - these are our baseline variances
            self._var_initial = self.gp_calculator.get_parameter_variance()

        # Store current variances
        self._var_current = self.gp_calculator.get_parameter_variance()

        self._fitted = True

        return self

    def _compute_dynamic_lambda(self) -> float:
        """
        Compute dynamic interaction weight λ_t based on current variance reduction.

        λ_t = λ_min                                                      if r_t > τ_1
              λ_min + (λ_max - λ_min) * (τ_1 - r_t) / (τ_1 - τ_2)      if τ_2 ≤ r_t ≤ τ_1
              λ_max                                                      if r_t < τ_2

        where r_t is the relative variance (current / initial) averaged over main effects.
        """
        if not self._fitted:
            return self.lambda_min

        # Get main effect variances
        offset = 1 if self.gp_calculator.include_intercept else 0
        n_main = self._n_features

        var_current_main = self._var_current[offset : offset + n_main]
        var_initial_main = self._var_initial[offset : offset + n_main]

        # Compute relative variance r_t
        # Avoid division by zero
        valid_mask = var_initial_main > 1e-10
        if not np.any(valid_mask):
            r_t = 1.0
        else:
            relative_vars = var_current_main[valid_mask] / var_initial_main[valid_mask]
            r_t = np.mean(relative_vars)

        # Apply piecewise linear function
        if r_t > self.tau_1:
            lambda_t = self.lambda_min
        elif r_t < self.tau_2:
            lambda_t = self.lambda_max
        else:
            # Linear interpolation
            lambda_t = self.lambda_min + (self.lambda_max - self.lambda_min) * (
                self.tau_1 - r_t
            ) / (self.tau_1 - self.tau_2)

        return lambda_t

    def _compute_info_gain(self, X_candidates: np.ndarray) -> np.ndarray:
        """
        Compute information gain term α_info for candidate points.

        α_info(x) = (1/|J|) Σ_j ΔVar[θ_j] + λ_t * (1/|I|) Σ_{j,k} ΔVar[θ_jk]
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        n_candidates = X_candidates.shape[0]
        info_scores = np.zeros(n_candidates)

        # Compute dynamic lambda
        lambda_t = self._compute_dynamic_lambda()

        for i in range(n_candidates):
            x = X_candidates[i : i + 1]  # Keep 2D shape

            # Compute variance reduction
            main_var_red, inter_var_red = self.gp_calculator.compute_variance_reduction(
                x
            )

            # Average main effect variance reduction
            if len(main_var_red) > 0:
                avg_main_var_red = np.mean(main_var_red)
            else:
                avg_main_var_red = 0.0

            # Average interaction effect variance reduction
            if len(inter_var_red) > 0:
                avg_inter_var_red = np.mean(inter_var_red)
            else:
                avg_inter_var_red = 0.0

            # Combine with dynamic weight
            info_scores[i] = avg_main_var_red + lambda_t * avg_inter_var_red

        return info_scores

    def _compute_coverage(self, X_candidates: np.ndarray) -> np.ndarray:
        """
        Compute spatial coverage term α_cov for candidate points.

        α_cov(x) = γ * COV(x; D_t)

        where COV is the minimum Gower distance to existing samples.
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Compute coverage scores
        coverage_scores = compute_coverage_batch(
            X_candidates,
            self._X_train,
            variable_types=self.variable_types,
            ranges=None,  # Will be computed automatically
            method=self.coverage_method,
        )

        return self.gamma * coverage_scores

    @t_batch_mode_transform()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate acquisition function on candidate points (BoTorch interface).

        Parameters
        ----------
        X : torch.Tensor
            Candidate points, shape (batch_size, q, d) or (batch_size, d)

        Returns
        -------
        torch.Tensor
            Acquisition scores, shape (batch_size,)
        """
        # Extract training data from model
        if hasattr(self.model, "train_inputs") and hasattr(self.model, "train_targets"):
            if self._X_train is None:
                # First call - extract and fit
                X_train_tensor = self.model.train_inputs[0]
                y_train_tensor = self.model.train_targets

                # Convert to numpy
                X_train = X_train_tensor.cpu().detach().numpy()
                y_train = y_train_tensor.cpu().detach().numpy()

                # Fit the model
                self.fit(X_train, y_train)

        if not self._fitted:
            # If still not fitted, return zeros
            return torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)

        # Convert X to numpy for computation
        X_np = X.cpu().detach().numpy()

        # Handle batch dimensions
        if X_np.ndim == 3:  # batch_size x q x d
            # For q > 1, evaluate each point and sum
            batch_size, q, d = X_np.shape
            X_np = X_np.reshape(-1, d)

        # Compute scores
        scores = self._evaluate_numpy(X_np)

        # Convert back to torch
        scores_tensor = torch.from_numpy(scores).to(dtype=X.dtype, device=X.device)

        # Reshape if needed
        if X.ndim == 3:
            scores_tensor = scores_tensor.reshape(batch_size, q).sum(dim=-1)

        return scores_tensor

    def _evaluate_numpy(self, X_candidates: np.ndarray) -> np.ndarray:
        """
        Evaluate acquisition function on numpy arrays (internal).

        Parameters
        ----------
        X_candidates : np.ndarray
            Candidate points (n_candidates, n_features)

        Returns
        -------
        np.ndarray
            Acquisition scores (n_candidates,)
        """
        if X_candidates.ndim == 1:
            X_candidates = X_candidates.reshape(1, -1)

        # Compute components
        info_scores = self._compute_info_gain(X_candidates)
        coverage_scores = self._compute_coverage(X_candidates)

        # Total acquisition score
        total_scores = info_scores + coverage_scores
        return total_scores

    def __call__(
        self, X_candidates: np.ndarray, return_components: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Evaluate acquisition function for candidate points (numpy interface).

        Parameters
        ----------
        X_candidates : np.ndarray
            Candidate points to evaluate (n_candidates, n_features)
        return_components : bool, default=False
            If True, return (total_scores, info_scores, coverage_scores)

        Returns
        -------
        scores : np.ndarray
            Acquisition scores (n_candidates,)
        info_scores : np.ndarray, optional
            Information gain scores (returned if return_components=True)
        coverage_scores : np.ndarray, optional
            Coverage scores (returned if return_components=True)

        Examples
        --------
        >>> acq_fn = VarianceReductionWithCoverageAcqf(model)
        >>> acq_fn.fit(X_train, y_train)
        >>> scores = acq_fn(X_candidates)
        >>> best_idx = np.argmax(scores)
        """
        # 当从BoTorch/AEPsych Server调用时,model会已经拟合
        # 检查model是否有训练数据并自动提取数据
        if hasattr(self.model, "train_inputs") and self.model.train_inputs is not None:
            if not self._fitted:
                # 从model提取训练数据用于计算
                X_train_tensor = self.model.train_inputs[0]
                y_train_tensor = self.model.train_targets
                X_train = X_train_tensor.cpu().detach().numpy()
                y_train = y_train_tensor.cpu().detach().numpy()
                # 调用fit来更新内部状态
                self.fit(X_train, y_train)
        elif not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if X_candidates.ndim == 1:
            X_candidates = X_candidates.reshape(1, -1)

        # Compute components
        info_scores = self._compute_info_gain(X_candidates)
        coverage_scores = self._compute_coverage(X_candidates)

        # Total acquisition score
        total_scores = info_scores + coverage_scores

        # Convert to torch tensor for BoTorch compatibility
        # Note: Our function is computed via numpy (no autodiff), so requires_grad=False
        import torch

        if not isinstance(total_scores, torch.Tensor):
            total_scores = torch.tensor(
                total_scores, dtype=torch.float32, requires_grad=False
            )

        if return_components:
            if not isinstance(info_scores, torch.Tensor):
                info_scores = torch.tensor(info_scores, dtype=torch.float32)
            if not isinstance(coverage_scores, torch.Tensor):
                coverage_scores = torch.tensor(coverage_scores, dtype=torch.float32)
            return total_scores, info_scores, coverage_scores

        return total_scores

    def select_next(
        self, X_candidates: np.ndarray, n_select: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select the next points to sample based on acquisition scores.

        Parameters
        ----------
        X_candidates : np.ndarray
            Candidate points (n_candidates, n_features)
        n_select : int, default=1
            Number of points to select

        Returns
        -------
        selected_X : np.ndarray
            Selected points (n_select, n_features)
        selected_indices : np.ndarray
            Indices of selected points in X_candidates (n_select,)

        Examples
        --------
        >>> acq_fn = VarianceReductionWithCoverageAcqf()
        >>> acq_fn.fit(X_train, y_train)
        >>> next_X, indices = acq_fn.select_next(X_candidates, n_select=5)
        """
        scores = self(X_candidates)

        # Select top n_select points
        selected_indices = np.argsort(scores)[-n_select:][::-1]
        selected_X = X_candidates[selected_indices]

        return selected_X, selected_indices

    def get_current_lambda(self) -> float:
        """
        Get the current value of dynamic interaction weight λ_t.

        Returns
        -------
        float
            Current λ_t value
        """
        return self._compute_dynamic_lambda()

    def get_variance_reduction_ratio(self) -> float:
        """
        Get the current variance reduction ratio r_t.

        Returns
        -------
        float
            Current r_t value (relative variance)
        """
        if not self._fitted:
            return 1.0

        offset = 1 if self.gp_calculator.include_intercept else 0
        n_main = self._n_features

        var_current_main = self._var_current[offset : offset + n_main]
        var_initial_main = self._var_initial[offset : offset + n_main]

        valid_mask = var_initial_main > 1e-10
        if not np.any(valid_mask):
            return 1.0

        relative_vars = var_current_main[valid_mask] / var_initial_main[valid_mask]
        return np.mean(relative_vars)
