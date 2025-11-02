#!/usr/bin/env python
"""
Test suite for EURAnovaPairAcqf
"""

import torch
import numpy as np
from eur_anova_pair_acquisition import EURAnovaPairAcqf
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import GaussianLikelihood


def test_basic():
    """Basic initialization and import test"""
    print("Test 1: Basic initialization")
    X_train = torch.randn(10, 3)
    y_train = torch.randn(10)
    model = SingleTaskGP(
        X_train, y_train.unsqueeze(-1), likelihood=GaussianLikelihood()
    )

    acqf = EURAnovaPairAcqf(
        model=model,
        interaction_pairs=[(0, 1)],
        variable_types={0: "continuous", 1: "categorical"},
    )
    print(f"  main_weight = {acqf.main_weight}")
    print(f"  lambda_max = {acqf.lambda_max}")
    print("  ✅ Test passed\n")


def test_forward():
    """Forward pass test"""
    print("Test 2: Forward pass")
    torch.manual_seed(42)
    X_train = torch.randn(15, 3)
    y_train = torch.randn(15)
    model = SingleTaskGP(
        X_train, y_train.unsqueeze(-1), likelihood=GaussianLikelihood()
    )
    model.eval()

    acqf = EURAnovaPairAcqf(
        model=model,
        interaction_pairs=[(0, 1)],
        variable_types={0: "continuous", 1: "integer", 2: "categorical"},
    )

    X_test = torch.randn(5, 1, 3)
    with torch.no_grad():
        acq_values = acqf(X_test)

    print(f"  Acq values shape: {acq_values.shape}")
    print(f"  Acq values range: [{acq_values.min():.4f}, {acq_values.max():.4f}]")
    print("  ✅ Test passed\n")


if __name__ == "__main__":
    print("=" * 50)
    print("EURAnovaPairAcqf Test Suite")
    print("=" * 50 + "\n")

    test_basic()
    test_forward()

    print("=" * 50)
    print("All tests completed successfully!")
    print("=" * 50)
