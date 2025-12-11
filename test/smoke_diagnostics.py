#!/usr/bin/env python3
"""Small smoke test for DiagnosticsManager logging behavior."""
from loguru import logger
import torch
from extensions.dynamic_eur_acquisition.modules.diagnostics import DiagnosticsManager
from extensions.dynamic_eur_acquisition.logging_setup import configure_logger

# Configure logger using our UTF-8 wrapper for Windows GBK compatibility
configure_logger(level="DEBUG")

if __name__ == "__main__":
    diagm = DiagnosticsManager(
        debug_components=True, enabled=True, verbose_mode=True, output_file=None
    )
    # create small tensors
    main = torch.tensor([0.1, 0.2, 0.3])
    pair = torch.tensor([0.01, 0.02])
    info = torch.tensor([0.5, 0.6])
    cov = torch.tensor([0.05, 0.07])
    diagm.update_effects(
        main_sum=main, pair_sum=pair, triplet_sum=None, info_raw=info, cov=cov
    )
    diag = diagm.get_diagnostics(
        lambda_t=0.5, gamma_t=0.2, lambda_2=0.4, n_train=10, fitted=True
    )
    # show both summary and verbose
    diagm.print_diagnostics(diag, verbose=False)
    diagm.print_diagnostics(diag, verbose=True)
    print("\nSmoke diagnostics test completed.")
