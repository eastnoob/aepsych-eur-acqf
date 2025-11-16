#!/usr/bin/env python3
"""Test old version"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from botorch.models import SingleTaskGP
from eur_anova_pair import EURAnovaPairAcqf

# Test with old version
model = SingleTaskGP(
    torch.rand(5, 3, dtype=torch.float64),
    torch.randn(5, 1, dtype=torch.float64)
)
model.eval()

acqf = EURAnovaPairAcqf(model, interaction_pairs=[(0, 1)])

X_candidates = torch.rand(10, 3, dtype=torch.float64)
scores = acqf(X_candidates)

print(f"Old version works! Shape: {scores.shape}")
