"""
Gower distance calculation for mixed-type variables.

Gower distance is a similarity measure that handles continuous, categorical,
and binary variables in a unified framework.
"""

import numpy as np
from typing import Dict, List, Optional, Union


def gower_distance(
    x1: np.ndarray,
    x2: np.ndarray,
    variable_types: Optional[Dict[int, str]] = None,
    ranges: Optional[Dict[int, float]] = None,
) -> float:
    """
    Calculate Gower distance between two data points.

    The Gower distance is defined as:
    d(x1, x2) = (1/p) * Σ_j δ_j(x1_j, x2_j)

    where δ_j is the dissimilarity for feature j:
    - For continuous: |x1_j - x2_j| / range_j
    - For categorical: 0 if x1_j == x2_j, else 1

    Parameters
    ----------
    x1 : np.ndarray
        First data point (1D array)
    x2 : np.ndarray
        Second data point (1D array)
    variable_types : Dict[int, str], optional
        Dictionary mapping feature index to type ('continuous' or 'categorical').
        If None, all features are assumed continuous.
    ranges : Dict[int, float], optional
        Dictionary mapping continuous feature index to its range.
        If None, ranges are computed from the data.

    Returns
    -------
    float
        Gower distance between x1 and x2 (in [0, 1])

    Examples
    --------
    >>> x1 = np.array([0.5, 0.3, 1.0])
    >>> x2 = np.array([0.7, 0.3, 2.0])
    >>> gower_distance(x1, x2)
    0.2
    """
    # Convert torch tensors to numpy
    import torch

    if isinstance(x1, torch.Tensor):
        x1 = x1.detach().cpu().numpy()
    if isinstance(x2, torch.Tensor):
        x2 = x2.detach().cpu().numpy()

    # Flatten to 1D if needed
    x1 = np.atleast_1d(x1).flatten()
    x2 = np.atleast_1d(x2).flatten()

    if x1.shape != x2.shape:
        raise ValueError(
            f"x1 and x2 must have the same shape, got {x1.shape} and {x2.shape}"
        )

    p = len(x1)

    # Default: all features are continuous
    if variable_types is None:
        variable_types = {i: "continuous" for i in range(p)}

    # Default: compute ranges from max-min
    if ranges is None:
        ranges = {}

    total_distance = 0.0
    valid_features = 0

    for j in range(p):
        var_type = variable_types.get(j, "continuous")

        if var_type == "continuous":
            # Handle missing values
            if np.isnan(x1[j]) or np.isnan(x2[j]):
                continue

            # Get range for normalization
            if j in ranges and ranges[j] > 0:
                range_j = ranges[j]
            else:
                # If range not provided, assume [0, 1] or use absolute difference
                range_j = 1.0

            distance_j = abs(x1[j] - x2[j]) / range_j
            total_distance += distance_j
            valid_features += 1

        elif var_type == "categorical":
            # Categorical: 0 if equal, 1 if different
            if x1[j] == x2[j]:
                distance_j = 0.0
            else:
                distance_j = 1.0
            total_distance += distance_j
            valid_features += 1
        else:
            raise ValueError(f"Unknown variable type: {var_type}")

    if valid_features == 0:
        return 0.0

    return total_distance / valid_features


def gower_distance_matrix(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    variable_types: Optional[Dict[int, str]] = None,
    ranges: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """
    Compute pairwise Gower distance matrix.

    Parameters
    ----------
    X : np.ndarray
        First set of data points (n_samples, n_features)
    Y : np.ndarray, optional
        Second set of data points (m_samples, n_features).
        If None, compute pairwise distances within X.
    variable_types : Dict[int, str], optional
        Dictionary mapping feature index to type.
    ranges : Dict[int, float], optional
        Dictionary mapping continuous feature index to its range.

    Returns
    -------
    np.ndarray
        Distance matrix of shape (n_samples, m_samples) or (n_samples, n_samples)

    Examples
    --------
    >>> X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    >>> D = gower_distance_matrix(X)
    >>> D.shape
    (3, 3)
    """
    if Y is None:
        Y = X
        symmetric = True
    else:
        symmetric = False

    n = X.shape[0]
    m = Y.shape[0]

    # Compute ranges from X if not provided
    if ranges is None:
        ranges = {}
        if variable_types is None:
            variable_types = {i: "continuous" for i in range(X.shape[1])}

        for j in range(X.shape[1]):
            if variable_types.get(j, "continuous") == "continuous":
                col = X[:, j]
                valid_vals = col[~np.isnan(col)]
                if len(valid_vals) > 0:
                    ranges[j] = max(valid_vals.max() - valid_vals.min(), 1e-10)

    D = np.zeros((n, m))

    for i in range(n):
        start_j = i if symmetric else 0
        for j in range(start_j, m):
            dist = gower_distance(X[i], Y[j], variable_types, ranges)
            D[i, j] = dist
            if symmetric and i != j:
                D[j, i] = dist

    return D


def compute_coverage(
    x: np.ndarray,
    X_sampled: np.ndarray,
    variable_types: Optional[Dict[int, str]] = None,
    ranges: Optional[Dict[int, float]] = None,
    method: str = "min_distance",
) -> float:
    """
    Compute spatial coverage score for a candidate point.

    Higher scores indicate the point is farther from existing samples,
    thus providing better coverage of the design space.

    Parameters
    ----------
    x : np.ndarray
        Candidate point (1D array)
    X_sampled : np.ndarray
        Already sampled points (n_samples, n_features)
    variable_types : Dict[int, str], optional
        Dictionary mapping feature index to type.
    ranges : Dict[int, float], optional
        Dictionary mapping continuous feature index to its range.
    method : str, default='min_distance'
        Method to compute coverage:
        - 'min_distance': Minimum distance to existing samples
        - 'mean_distance': Mean distance to existing samples
        - 'median_distance': Median distance to existing samples

    Returns
    -------
    float
        Coverage score (higher is better)

    Examples
    --------
    >>> x = np.array([0.9, 0.9])
    >>> X_sampled = np.array([[0.1, 0.1], [0.2, 0.2]])
    >>> compute_coverage(x, X_sampled)
    0.8
    """
    if X_sampled.shape[0] == 0:
        # No samples yet, maximum coverage
        return 1.0

    # Compute distances from x to all sampled points
    distances = np.array(
        [
            gower_distance(x, X_sampled[i], variable_types, ranges)
            for i in range(X_sampled.shape[0])
        ]
    )

    if method == "min_distance":
        coverage = np.min(distances)
    elif method == "mean_distance":
        coverage = np.mean(distances)
    elif method == "median_distance":
        coverage = np.median(distances)
    else:
        raise ValueError(f"Unknown coverage method: {method}")

    return coverage


def compute_coverage_batch(
    X_candidates: np.ndarray,
    X_sampled: np.ndarray,
    variable_types: Optional[Dict[int, str]] = None,
    ranges: Optional[Dict[int, float]] = None,
    method: str = "min_distance",
) -> np.ndarray:
    """
    Compute spatial coverage scores for a batch of candidate points.

    Parameters
    ----------
    X_candidates : np.ndarray
        Candidate points (n_candidates, n_features)
    X_sampled : np.ndarray
        Already sampled points (n_samples, n_features)
    variable_types : Dict[int, str], optional
        Dictionary mapping feature index to type.
    ranges : Dict[int, float], optional
        Dictionary mapping continuous feature index to its range.
    method : str, default='min_distance'
        Method to compute coverage.

    Returns
    -------
    np.ndarray
        Coverage scores for all candidates (n_candidates,)

    Examples
    --------
    >>> X_candidates = np.array([[0.8, 0.8], [0.9, 0.9]])
    >>> X_sampled = np.array([[0.1, 0.1], [0.2, 0.2]])
    >>> compute_coverage_batch(X_candidates, X_sampled)
    array([0.7, 0.8])
    """
    return np.array(
        [
            compute_coverage(X_candidates[i], X_sampled, variable_types, ranges, method)
            for i in range(X_candidates.shape[0])
        ]
    )
