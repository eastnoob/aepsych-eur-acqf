"""
Diagnostics manager for EUR acquisition — clean, single-copy module.

Provides controlled runtime diagnostics output using `loguru`.

Behaviors:
- `enabled` (INFO): concise per-forward summary
- `verbose_mode` / `verbose=True` (DEBUG): detailed arrays and optional file dump
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import torch
from loguru import logger
from pathlib import Path
from datetime import datetime


class DiagnosticsManager:
    """Diagnostics manager for EUR acquisition functions.

    Args:
        debug_components: keep per-component caches for inspection
        enabled: emit concise INFO summaries when True
        verbose_mode: always emit DEBUG-level detailed diagnostics when True
        output_file: optional path; when set verbose output is written there
    """

    def __init__(
        self,
        debug_components: bool = False,
        enabled: bool = False,
        verbose_mode: bool = False,
        output_file: Optional[str] = None,
    ) -> None:
        self.debug_components = debug_components
        self.enabled = enabled
        self.verbose_mode = verbose_mode
        self.output_file = Path(output_file) if output_file else None
        self.history = []

        # last-seen effect tensors (kept on CPU)
        self._last_main: Optional[torch.Tensor] = None
        self._last_pair: Optional[torch.Tensor] = None
        self._last_triplet: Optional[torch.Tensor] = None
        self._last_info: Optional[torch.Tensor] = None
        self._last_cov: Optional[torch.Tensor] = None

    def update_effects(
        self,
        main_sum: Optional[torch.Tensor] = None,
        pair_sum: Optional[torch.Tensor] = None,
        triplet_sum: Optional[torch.Tensor] = None,
        info_raw: Optional[torch.Tensor] = None,
        cov: Optional[torch.Tensor] = None,
    ) -> None:
        """Cache the most recent effect tensors (safely moved to CPU).

        This method is defensive: when an input is missing we maintain a
        sensible zero-shaped tensor so downstream diagnostics remain robust.
        """
        # pick a template from provided tensors to determine shape
        template = None
        for t in (main_sum, pair_sum, triplet_sum, info_raw, cov):
            if t is not None:
                try:
                    template = t.detach().cpu()
                    break
                except Exception:
                    template = None

        if template is None:
            template = torch.zeros(1, dtype=torch.float32)

        def _safe_to_cpu(x: Optional[torch.Tensor]) -> torch.Tensor:
            if x is None:
                try:
                    return torch.zeros_like(template)
                except Exception:
                    return torch.zeros(1, dtype=torch.float32)
            try:
                return x.detach().cpu()
            except Exception:
                t = torch.as_tensor(x)
                if t.numel() == 1 and template.numel() > 1:
                    return torch.zeros_like(template)
                return t.detach().cpu()

        self._last_main = _safe_to_cpu(main_sum)
        self._last_pair = _safe_to_cpu(pair_sum)
        self._last_triplet = _safe_to_cpu(triplet_sum)
        self._last_info = _safe_to_cpu(info_raw)
        self._last_cov = _safe_to_cpu(cov)

    def get_diagnostics(
        self,
        lambda_t: float,
        gamma_t: float,
        lambda_2: Optional[float] = None,
        lambda_3: Optional[float] = None,
        n_train: int = 0,
        fitted: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return a diagnostics dictionary ready for logging or writing."""
        # 1. 先从配置初始化（静态参数）
        diag: Dict[str, Any] = {}
        if config is not None:
            diag.update(config)

        # 2. 用运行时动态值覆盖（确保诊断信息反映当前真实状态）
        diag.update({
            "lambda_t": lambda_t,
            "lambda_2": lambda_2 if lambda_2 is not None else lambda_t,
            "lambda_3": lambda_3 if lambda_3 is not None else 0.0,
            "gamma_t": gamma_t,
            "n_train": n_train,
            "fitted": fitted,
        })

        if self._last_main is not None:
            diag["main_effects_sum"] = self._last_main
        if self._last_pair is not None:
            diag["pair_effects_sum"] = self._last_pair
        if self._last_triplet is not None:
            diag["triplet_effects_sum"] = self._last_triplet
        if self._last_info is not None:
            diag["info_raw"] = self._last_info
        if self._last_cov is not None:
            diag["coverage"] = self._last_cov

        return diag

    def print_diagnostics(self, diag: Dict[str, Any], verbose: bool = False) -> None:
        """Emit diagnostics via `loguru` according to configured levels.

        - When `enabled` is True: an INFO summary is emitted.
        - When `verbose` or `verbose_mode` is True: DEBUG-level details
          (including array summaries) are emitted and, if `output_file`
          is set, a full dump is written to that file (timestamped).
        """
        # Store in history for table view
        n_train = diag.get("n_train", 0)
        r_t = diag.get("r_t", None)
        lambda2 = diag.get("lambda_2", diag.get("lambda_t", 0.0))
        gamma = diag.get("gamma_t", 0.0)
        fitted = diag.get("fitted", False)

        # Only add if n_train is new or history is empty
        if not self.history or self.history[-1]["n"] != n_train:
            self.history.append(
                {
                    "n": n_train,
                    "r_t": r_t,
                    "λ_2": lambda2,
                    "γ": gamma,
                    "fitted": fitted,
                }
            )

        if not self.enabled and not verbose and not self.verbose_mode:
            return

        try:
            summary = f"[EUR] Trial {n_train:2d} | r_t={ (f'{r_t:.3f}') if r_t is not None else 'N/A':5s} | λ_2={lambda2:.3f} | γ={gamma:.3f} | fitted={int(bool(fitted))}"
        except Exception:
            summary = "[EUR] diagnostics summary unavailable"

        if self.enabled:
            logger.debug(summary)

        if verbose or self.verbose_mode:
            logger.debug("--- EUR Detailed Diagnostics BEGIN ---")
            logger.debug(
                f"lambda_2={diag.get('lambda_2', None)} lambda_3={diag.get('lambda_3', None)}"
            )
            logger.debug(
                f"gamma_t={diag.get('gamma_t', None)} n_train={diag.get('n_train', None)}"
            )

            if "main_effects_sum" in diag:
                main = diag["main_effects_sum"]
                logger.debug(
                    f"main: mean={main.mean():.6f} std={main.std():.6f} shape={tuple(main.shape)}"
                )
            if "pair_effects_sum" in diag:
                pair = diag["pair_effects_sum"]
                logger.debug(
                    f"pair: mean={pair.mean():.6f} std={pair.std():.6f} shape={tuple(pair.shape)}"
                )
            if "triplet_effects_sum" in diag:
                triplet = diag["triplet_effects_sum"]
                logger.debug(
                    f"triplet: mean={triplet.mean():.6f} std={triplet.std():.6f} shape={tuple(triplet.shape)}"
                )

            if self.output_file is not None:
                try:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    outpath = self.output_file.with_name(
                        self.output_file.stem + f"_{ts}" + self.output_file.suffix
                    )
                    with open(outpath, "w", encoding="utf-8") as fh:
                        fh.write("EUR Detailed Diagnostics\n")
                        for k, v in diag.items():
                            fh.write(f"{k}: {repr(v)}\n")
                    logger.info(f"Wrote detailed diagnostics to {outpath}")
                except Exception as e:
                    logger.error(
                        f"Failed to write diagnostics to {self.output_file}: {e}"
                    )

            logger.debug("--- EUR Detailed Diagnostics END ---")

        # Human-friendly effect summaries at DEBUG when enabled
        if "main_effects_sum" in diag:
            main = diag["main_effects_sum"]
            logger.debug("\n【效应贡献】(最后一次 forward() 调用)")
            logger.debug(f"  主效应总和: mean={main.mean():.4f}, std={main.std():.4f}")

            if "pair_effects_sum" in diag:
                pair = diag["pair_effects_sum"]
                logger.debug(
                    f"  二阶交互总和: mean={pair.mean():.4f}, std={pair.std():.4f}"
                )

            if "triplet_effects_sum" in diag:
                triplet = diag["triplet_effects_sum"]
                logger.debug(
                    f"  三阶交互总和: mean={triplet.mean():.4f}, std={triplet.std():.4f}"
                )

            if "info_raw" in diag:
                info = diag["info_raw"]
                logger.debug(f"  信息项: mean={info.mean():.4f}, std={info.std():.4f}")

            if "coverage" in diag:
                cov = diag["coverage"]
                logger.debug(f"  覆盖项: mean={cov.mean():.4f}, std={cov.std():.4f}")

            if verbose:
                logger.debug(f"\n  主效应数组:\n    {main}")
                if "pair_effects_sum" in diag:
                    logger.debug(f"  二阶交互数组:\n    {pair}")
                if "triplet_effects_sum" in diag:
                    logger.debug(f"  三阶交互数组:\n    {triplet}")
        else:
            logger.warning(
                "\n⚠️  效应贡献数据不可用 - 提示: 初始化时设置 debug_components=True"
            )

    def save_history_csv(self, path: str | Path) -> None:
        """Save the weight history to a CSV file."""
        if not self.history:
            return
            
        import pandas as pd
        try:
            df = pd.DataFrame(self.history)
            # Rename columns for clarity in CSV
            df.columns = ['trial', 'r_t', 'lambda_2', 'gamma', 'fitted']
            df.to_csv(path, index=False)
            logger.info(f"[EUR] Saved weight history to {path}")
        except Exception as e:
            logger.error(f"[EUR] Failed to save history to {path}: {e}")
