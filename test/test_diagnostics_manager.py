import re
import os
from loguru import logger
import torch
import pytest

from extensions.dynamic_eur_acquisition.modules.diagnostics import DiagnosticsManager


def _collect_logs(func):
    """Helper: add a temporary sink that collects formatted log lines."""
    records = []

    def sink(line):
        # line is the formatted string
        records.append(line)

    sink_id = logger.add(sink, level="DEBUG", format="{level} {message}")
    try:
        func(records)
    finally:
        logger.remove(sink_id)

    return records


def test_disabled_no_output():
    dm = DiagnosticsManager(enabled=False, verbose_mode=False)
    dm.update_effects(
        main_sum=torch.tensor([1.0, 2.0]),
        pair_sum=torch.tensor([0.1, 0.2]),
        info_raw=torch.tensor([0.5, 0.6]),
        cov=torch.tensor([0.05, 0.02]),
    )

    diag = dm.get_diagnostics(lambda_t=0.3, gamma_t=0.1, n_train=5, fitted=True)

    def run(records):
        dm.print_diagnostics(diag, verbose=False)

    records = _collect_logs(run)

    # When disabled, no summary line starting with [EUR] should be emitted
    assert not any("[EUR] n_train" in r for r in records)


def test_enabled_info_output():
    dm = DiagnosticsManager(enabled=True, verbose_mode=False)
    dm.update_effects(main_sum=torch.tensor([3.0, 4.0]))
    diag = dm.get_diagnostics(lambda_t=0.5, gamma_t=0.2, n_train=10, fitted=False)

    def run(records):
        dm.print_diagnostics(diag, verbose=False)

    records = _collect_logs(run)

    # Should contain an INFO summary line with n_train and λ
    assert any(re.search(r"\[EUR\] n_train=10 .*λ₂=0\.5", r) for r in records)


def test_verbose_writes_file(tmp_path):
    outdir = tmp_path
    outfile = outdir / "diag_output.txt"
    dm = DiagnosticsManager(enabled=True, verbose_mode=True, output_file=str(outfile))

    dm.update_effects(
        main_sum=torch.tensor([0.3, 0.7, 1.2]),
        pair_sum=torch.tensor([0.01, 0.02]),
    )

    diag = dm.get_diagnostics(lambda_t=0.2, gamma_t=0.05, n_train=2, fitted=True)

    def run(records):
        dm.print_diagnostics(diag, verbose=True)

    _ = _collect_logs(run)

    # Verify that a file with the diagnostics was written
    files = list(outdir.iterdir())
    assert any(f.name.startswith("diag_output") for f in files)
