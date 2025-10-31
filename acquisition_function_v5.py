"""
EUR Acquisition Function (V5): Ordinal-safe info with torch posterior + numpy Gower coverage.

Design goals:
- Keep regression/continuous behavior intact (can reuse linear-ΔVar path).
- For ordinal models, compute information from posterior latent stats in torch:
  - Use posterior variance Var[f(x)] as base uncertainty.
  - Optional ordinal-aware terms: entropy over categories (via Normal-CDF and cutpoints),
    and a cutpoint sensitivity weight (closer to thresholds → higher weight).
- Coverage uses existing numpy Gower implementation; only coverage path converts to numpy.
- Robust to batch/q shapes, device/dtype, and transform/canonicalization.

Notes:
- This class is intended to coexist with V4. V4 remains available; V5 adds ordinal compatibility.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import gpytorch
from botorch.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform

try:
    from .gower_distance import compute_coverage_batch
    from .gp_variance import GPVarianceCalculator
except Exception:  # pragma: no cover
    from gower_distance import compute_coverage_batch  # type: ignore
    from gp_variance import GPVarianceCalculator  # type: ignore


EPS = 1e-8


class EURAcqfV5(AcquisitionFunction):
    def __init__(
        self,
        model: Model,
        # Coverage weight
        gamma: float = 0.3,
        # For regression path (ΔVar linear path), keep the original knobs
        lambda_min: float = 0.2,
        lambda_max: float = 2.0,
        tau_1: float = 0.5,
        tau_2: float = 0.1,
        interaction_terms: Optional[List[Tuple[int, int]]] = None,
        noise_variance: float = 1.0,
        prior_variance: float = 1.0,
        variable_types: Optional[Dict[int, str]] = None,
        coverage_method: str = "min_distance",
        # Allow list-style variable types mapping (e.g., ["categorical", "integer", ...])
        variable_types_list: Optional[Union[List[str], str]] = None,
        # Ordinal info term blend weights (sum not necessarily 1; will be re-normalized per batch)
        use_entropy: bool = True,
        use_cut_sensitivity: bool = True,
        debug_components: Union[bool, str] = False,
    ) -> None:
        super().__init__(model=model)

        self.gamma = gamma
        self.coverage_method = coverage_method

        # Keep regression/ΔVar path parameters for backward compatibility
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.tau_1 = tau_1
        self.tau_2 = tau_2

        self.noise_variance = noise_variance
        self.prior_variance = prior_variance

        # Linear ΔVar calculator (used for regression path only)
        self.gp_calculator = GPVarianceCalculator(
            noise_variance=noise_variance,
            prior_variance=prior_variance,
            include_intercept=True,
        )

        # Runtime caches
        self._X_train_np: Optional[np.ndarray] = None
        self._y_train_np: Optional[np.ndarray] = None
        self._var_initial: Optional[np.ndarray] = None
        self._var_current: Optional[np.ndarray] = None
        self._n_features: Optional[int] = None
        self._fitted: bool = False
        self._last_hist_n: int = -1

        self.variable_types: Optional[Dict[int, str]] = variable_types
        if variable_types_list is not None and self.variable_types is None:
            tokens = (
                [
                    t.strip()
                    for t in str(variable_types_list)
                    .strip("()[]")
                    .replace(";", ",")
                    .split(",")
                    if t.strip()
                ]
                if isinstance(variable_types_list, str)
                else list(variable_types_list)
            )
            vt_map: Dict[int, str] = {}
            for i, t in enumerate(tokens):
                t_l = str(t).lower()
                if t_l.startswith("cat"):
                    vt_map[i] = "categorical"
                elif t_l.startswith("int"):
                    vt_map[i] = "integer"
            if len(vt_map) > 0:
                self.variable_types = vt_map

        if isinstance(debug_components, str):
            self.debug_components = debug_components.strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
        else:
            self.debug_components = bool(debug_components)
        self._last_info: Optional[torch.Tensor] = None
        self._last_cov: Optional[torch.Tensor] = None

        # Ordinal info switches
        self.use_entropy = use_entropy
        self.use_cut_sensitivity = use_cut_sensitivity
        self._interaction_terms = interaction_terms or []

    # ---------- transforms & type inference ----------
    def _get_param_transforms(self):
        try:
            return getattr(self.model, "transforms", None)
        except Exception:
            return None

    def _canonicalize_torch(self, X: torch.Tensor) -> torch.Tensor:
        """Apply model's parameter transforms to X in torch; return canonical X."""
        try:
            tf = self._get_param_transforms()
            if tf is None:
                return X
            return tf.transform(X)
        except Exception:
            return X

    def _maybe_infer_variable_types(self) -> None:
        if self.variable_types is not None:
            return
        tf = self._get_param_transforms()
        if tf is None:
            return
        vt: Dict[int, str] = {}
        try:
            import importlib

            mod = importlib.import_module("aepsych.transforms.ops")
            Categorical = getattr(mod, "Categorical")
            Round = getattr(mod, "Round")

            for sub in tf.values():
                if hasattr(sub, "indices") and isinstance(sub.indices, list):
                    for idx in sub.indices:
                        if isinstance(sub, Categorical):
                            vt[idx] = "categorical"
                        elif isinstance(sub, Round):
                            vt.setdefault(idx, "integer")
        except Exception:
            pass
        if len(vt) > 0:
            self.variable_types = vt

    # ---------- data sync & regression ΔVar helpers ----------
    def _fit_internal_np(self, X_np: np.ndarray, y_np: np.ndarray) -> None:
        self._X_train_np = X_np.copy()
        self._y_train_np = y_np.copy()
        self._n_features = X_np.shape[1]

        # Fit linear ΔVar calculator
        self.gp_calculator.fit(X_np, y_np, self._interaction_terms)

        if self._var_initial is None:
            self._var_initial = self.gp_calculator.get_parameter_variance()
        self._var_current = self.gp_calculator.get_parameter_variance()
        self._fitted = True

    def _ensure_fresh_data(self) -> None:
        if not hasattr(self.model, "train_inputs") or self.model.train_inputs is None:
            return
        X_t = self.model.train_inputs[0]
        y_t = getattr(self.model, "train_targets", None)
        if X_t is None or y_t is None:
            return
        n = X_t.shape[0]
        if (not self._fitted) or (n != self._last_hist_n):
            X_np = X_t.detach().cpu().numpy()
            y_np = y_t.detach().cpu().numpy()
            self._fit_internal_np(X_np, y_np)
            self._last_hist_n = n
            self._maybe_infer_variable_types()

    def _compute_dynamic_lambda(self) -> float:
        if not self._fitted or self._var_current is None or self._var_initial is None:
            return self.lambda_min
        offset = 1 if self.gp_calculator.include_intercept else 0
        n_main = self._n_features or 0
        var_cur = self._var_current[offset : offset + n_main]
        var_ini = self._var_initial[offset : offset + n_main]
        valid = var_ini > 1e-10
        if not np.any(valid):
            r_t = 1.0
        else:
            r_t = float(np.mean(var_cur[valid] / var_ini[valid]))
        if r_t > self.tau_1:
            return self.lambda_min
        if r_t < self.tau_2:
            return self.lambda_max
        return self.lambda_min + (self.lambda_max - self.lambda_min) * (
            (self.tau_1 - r_t) / (self.tau_1 - self.tau_2)
        )

    def _compute_info_gain_regression_np(self, X_can_np: np.ndarray) -> np.ndarray:
        assert self._fitted, "Call after data refresh"
        n = X_can_np.shape[0]
        scores = np.zeros(n, dtype=float)
        lam = self._compute_dynamic_lambda()
        for i in range(n):
            x = X_can_np[i : i + 1]
            main_red, inter_red = self.gp_calculator.compute_variance_reduction(x)
            if main_red is not None and len(main_red) > 0:
                main_red = np.maximum(main_red, 0.0)
            if inter_red is not None and len(inter_red) > 0:
                inter_red = np.maximum(inter_red, 0.0)
            avg_main = float(np.mean(main_red)) if len(main_red) > 0 else 0.0
            avg_inter = float(np.mean(inter_red)) if len(inter_red) > 0 else 0.0
            scores[i] = avg_main + lam * avg_inter
        return scores

    # ---------- ordinal-safe info in torch ----------
    def _is_ordinal(self) -> bool:
        """Robustly detect whether the model is ordinal.

        Heuristics:
        - Likelihood class name contains 'ordinal'
        - Likelihood has attributes like n_levels / num_levels
        - Cutpoints can be retrieved
        """
        try:
            lk = getattr(self.model, "likelihood", None)
            if lk is None:
                return False
            name = type(lk).__name__.lower()
            if "ordinal" in name:
                return True
            if hasattr(lk, "n_levels") or hasattr(lk, "num_levels"):
                return True
            # Last check: cutpoints existence
            c = self._get_cutpoints(device=torch.device("cpu"), dtype=torch.float64)
            return c is not None
        except Exception:
            return False

    def _get_cutpoints(
        self, device: torch.device, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        if not self._is_ordinal():
            return None
        cand_names = ["cutpoints", "cut_points", "cut_points_", "_cutpoints"]
        lk = self.model.likelihood
        for name in cand_names:
            c = getattr(lk, name, None)
            if c is not None:
                try:
                    c_t = torch.as_tensor(c, device=device, dtype=dtype)
                    return c_t.view(-1)
                except Exception:
                    continue
        return None

    @staticmethod
    def _normal_cdf(z: torch.Tensor) -> torch.Tensor:
        # Φ(z) = 0.5 * (1 + erf(z / sqrt(2)))
        return 0.5 * (
            1.0
            + torch.erf(
                z / torch.sqrt(torch.tensor(2.0, device=z.device, dtype=z.dtype))
            )
        )

    def _ordinal_entropy_from_mv(
        self, mean: torch.Tensor, var: torch.Tensor, cutpoints: torch.Tensor
    ) -> torch.Tensor:
        # mean: (B,), var: (B,), cutpoints: (K-1,)
        std = torch.sqrt(torch.clamp(var, min=EPS))
        # Build segment CDFs
        # For each cutpoint c_k, compute Φ((c_k - m)/s)
        z = (cutpoints.view(1, -1) - mean.view(-1, 1)) / std.view(-1, 1)
        cdfs = self._normal_cdf(z).clamp(EPS, 1 - EPS)  # (B, K-1)
        # probs: [p0, p1, ..., p_{K-1}] where K levels
        p0 = cdfs[:, :1]
        p_last = 1.0 - cdfs[:, -1:]
        if cdfs.shape[1] >= 2:
            mids = torch.clamp(cdfs[:, 1:] - cdfs[:, :-1], min=EPS)
            probs = torch.cat([p0, mids, p_last], dim=1)
        else:
            probs = torch.cat([p0, p_last], dim=1)
        probs = torch.clamp(probs, min=EPS, max=1.0)
        entropy = -torch.sum(probs * torch.log(probs), dim=-1)  # (B,)
        return entropy

    def _compute_info_gain_ordinal_torch(self, X_can_t: torch.Tensor) -> torch.Tensor:
        # Ensure correct device/dtype
        X_can_t = X_can_t.to(
            device=(
                self.model.mean_module.constant.device
                if hasattr(self.model, "mean_module")
                else X_can_t.device
            )
        )
        with torch.no_grad():
            posterior = self.model.posterior(X_can_t)
            mean = posterior.mean
            var = getattr(posterior, "variance", None)
            if var is None:
                # Try covariance diag
                try:
                    cov = posterior.variance  # some models expose variance alias
                    var = cov
                except Exception:
                    var = None

            # Reduce event dims → scalar per batch example
            def _reduce_event(x: torch.Tensor) -> torch.Tensor:
                # Try to squeeze event dims while keeping batch
                while x.dim() > 1 and x.shape[-1] == 1:
                    x = x.squeeze(-1)
                # If still multi-event, take mean over last dim
                if x.dim() > 1:
                    x = x.mean(dim=-1)
                return x.view(-1)

            mean_r = _reduce_event(mean)
            if var is None:
                # fall back to uniform info if variance is not available
                base_info = torch.ones_like(mean_r)
            else:
                var_r = _reduce_event(var)
                base_info = torch.clamp(var_r, min=EPS)

            info_terms: List[torch.Tensor] = [base_info]

            cutpoints = self._get_cutpoints(device=mean_r.device, dtype=mean_r.dtype)
            if cutpoints is not None:
                try:
                    if self.use_entropy:
                        ent = self._ordinal_entropy_from_mv(
                            mean_r, base_info, cutpoints
                        )
                        info_terms.append(ent)
                    if self.use_cut_sensitivity:
                        # distance to nearest cutpoint normalized by std
                        std = torch.sqrt(torch.clamp(base_info, min=EPS))
                        dists = torch.min(
                            torch.abs(mean_r.view(-1, 1) - cutpoints.view(1, -1)), dim=1
                        ).values
                        sens = torch.exp(-dists / std)
                        info_terms.append(sens)
                except Exception:
                    pass

            # Blend info terms by batch standardization then mean
            stacked = torch.stack(info_terms, dim=0)  # (T, B)
            # Standardize per-term across batch
            mu = stacked.mean(dim=1, keepdim=True)
            sigma = stacked.std(dim=1, keepdim=True)
            normed = (stacked - mu) / (sigma + EPS)
            info = normed.mean(dim=0)  # (B,)
            return info

    # ---------- coverage (numpy Gower) ----------
    def _compute_coverage_numpy(self, X_can_t: torch.Tensor) -> torch.Tensor:
        assert self._fitted and self._X_train_np is not None
        vt = None
        if self.variable_types is not None:
            vt = {
                k: ("categorical" if v == "categorical" else "continuous")
                for k, v in self.variable_types.items()
            }

        X_np = X_can_t.detach().cpu().numpy()
        cov_np = compute_coverage_batch(
            X_np,
            self._X_train_np,
            variable_types=vt,
            ranges=None,
            method=self.coverage_method,
        )
        cov_t = torch.from_numpy(cov_np).to(dtype=X_can_t.dtype, device=X_can_t.device)
        return cov_t

    # ---------- forward ----------
    @t_batch_mode_transform()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Sync data and types
        self._ensure_fresh_data()
        if (
            not self._fitted
            or self._X_train_np is None
            or (
                isinstance(self._X_train_np, np.ndarray)
                and self._X_train_np.shape[0] == 0
            )
        ):
            # No history yet: return random exploration to avoid degenerate ties
            B = X.shape[0] if X.dim() != 3 else X.shape[0]
            return torch.rand(B, dtype=X.dtype, device=X.device)

        # Flatten batch/q to (B, d)
        if X.dim() == 3:
            B, q, d = X.shape
            if q != 1:
                raise AssertionError(f"EURAcqfV5 currently supports q=1, got q={q}")
            X_flat = X.squeeze(1)
        else:
            B, d = X.shape
            X_flat = X

        # Canonicalize in torch
        X_can_t = self._canonicalize_torch(X_flat)

        # Info term: use ordinal-safe torch posterior path for all models
        try:
            info_t = self._compute_info_gain_ordinal_torch(X_can_t)
        except Exception:
            # Fallback to uniform info signal
            info_t = torch.ones(
                X_can_t.shape[0], dtype=X_can_t.dtype, device=X_can_t.device
            )

        # Coverage term via numpy Gower
        try:
            cov_t = self._compute_coverage_numpy(X_can_t)
        except Exception:
            cov_t = torch.zeros_like(info_t)

        # Per-batch standardization to similar scales
        def _stdz(x: torch.Tensor) -> torch.Tensor:
            mu = x.mean()
            sd = x.std()
            return (x - mu) / (sd + EPS)

        info_n = _stdz(info_t)
        cov_n = _stdz(cov_t)
        total = info_n + self.gamma * cov_n

        # Tie-breaker: add small jitter if near-constant across pool
        if (total.max() - total.min()) < 1e-9:
            total = total + (1e-3 * torch.rand_like(total))

        if self.debug_components:
            self._last_info = info_t.detach().cpu()
            self._last_cov = cov_t.detach().cpu()

        return total.view(B)
