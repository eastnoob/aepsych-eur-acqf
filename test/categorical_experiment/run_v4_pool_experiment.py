#!/usr/bin/env python3
"""
Small pool-based experiment using PoolBasedGenerator with selectable EUR acqf (V1/V3A/V4).
- Discrete mixed space: color(5) x layout(4) x font_size(12..22) x animation(3)
- Pool size ~120 (sampled)
- 15 init + 45 opt (total 60)
- Synthetic user rating in [0,10]
"""
import os
import sys
import random
from itertools import product

# Paths: add temp_aepsych and extensions
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "temp_aepsych"))
sys.path.insert(0, ROOT)

import argparse
import torch
import numpy as np
from typing import List, Tuple

from aepsych.models import GPRegressionModel
from extensions.custom_generators.pool_based_generator import PoolBasedGenerator
from extensions.dynamic_eur_acquisition import (
    EURAcqfV4,
    VarianceReductionWithCoverageAcqf,
    HardExclusionAcqf,
)


def make_pool(sample_n: int = 120, seed: int = 42) -> torch.Tensor:
    random.seed(seed)
    rng = np.random.default_rng(seed)
    colors = list(range(5))  # 0..4
    layouts = list(range(4))  # 0..3
    font_sizes = list(range(12, 23))  # 12..22
    anims = list(range(3))  # 0..2

    all_combos: List[Tuple[int, int, int, int]] = list(
        product(colors, layouts, font_sizes, anims)
    )
    rng.shuffle(all_combos)
    sel = all_combos[:sample_n]
    arr = np.array(sel, dtype=float)
    return torch.tensor(arr, dtype=torch.float64)


def user_score(x: torch.Tensor, noise: float = 0.25, seed: int = 0) -> float:
    """Simple synthetic rating in [0,10]. x: [4] in transformed discrete space."""
    # unpack
    c, l, f, a = x.tolist()
    # base preferences (arbitrary but reproducible)
    base = 0.0
    # color weight
    base += [1.0, 2.0, 1.5, 0.5, 0.8][int(c)]
    # layout weight
    base += [2.0, 1.0, 1.5, 0.7][int(l)]
    # font size preference peak around 20-22
    base += max(0.0, 2.5 - abs(f - 21.0) * 0.5)
    # animation
    base += [0.5, 1.2, 1.8][int(a)]
    # normalize to around [0,10]
    score = base * 1.6
    # add noise
    rng = np.random.default_rng(seed)
    score += float(rng.normal(0.0, noise))
    return float(np.clip(score, 0.0, 10.0))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pool-based EUR experiment (V1/V3A/V4)"
    )
    parser.add_argument(
        "--acqf",
        choices=["v1", "v3a", "v4"],
        default="v4",
        help="Which EUR version to use",
    )
    parser.add_argument(
        "--sample-n", type=int, default=120, help="Pool size (sampled from full grid)"
    )
    parser.add_argument(
        "--n-init", type=int, default=15, help="Initial random samples from pool"
    )
    parser.add_argument("--n-opt", type=int, default=45, help="Optimization iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    seed = args.seed
    acqf_choice = args.acqf.lower()
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build pool in transformed space (integers as floats)
    pool = make_pool(sample_n=args.sample_n, seed=seed)
    lb = torch.tensor([0.0, 0.0, 12.0, 0.0], dtype=torch.float64)
    ub = torch.tensor([4.0, 3.0, 22.0, 2.0], dtype=torch.float64)

    # Model
    model = GPRegressionModel(dim=4)

    # Init points from pool
    init_idx = torch.randperm(pool.shape[0])[: args.n_init]
    X_init = pool[init_idx]
    y_init = torch.tensor(
        [user_score(x, seed=seed + i) for i, x in enumerate(X_init)],
        dtype=torch.float64,
    )

    # Fit
    model.fit(X_init, y_init)

    # Choose acqf class
    acqf_map = {
        "v1": VarianceReductionWithCoverageAcqf,
        "v3a": HardExclusionAcqf,
        "v4": EURAcqfV4,
    }
    AcqfCls = acqf_map[acqf_choice]

    # Generator (Pool-based + chosen EUR)
    gen = PoolBasedGenerator(
        lb=lb,
        ub=ub,
        pool_points=pool,
        acqf=AcqfCls,
        acqf_kwargs={
            "lambda_min": 0.2,
            "lambda_max": 2.0,
            "tau_1": 0.5,
            "tau_2": 0.1,
            "gamma": 0.5,
            # variable_types 由 V4 自动从 transforms 推断（此处模型未包 transforms，离散值已是规范形）
        },
        allow_resampling=False,
        shuffle=False,
        seed=seed,
    )

    # Optimize iterations
    X_all = [x for x in X_init]
    y_all = [float(v) for v in y_init]

    for t in range(args.n_opt):
        x_next = gen.gen(num_points=1, model=model)[0]
        y_next = user_score(x_next, seed=1000 + t)
        # update model (warm-start)
        X_all.append(x_next)
        y_all.append(y_next)
        model.update(torch.stack(X_all), torch.tensor(y_all, dtype=torch.float64))

    # Report
    X_arr = torch.stack(X_all).detach().cpu().numpy()
    uniq = {tuple(map(float, row)) for row in X_arr.tolist()}
    print("Total trials:", len(X_all))
    print("Unique designs:", len(uniq))
    # quick distribution
    scores = np.array(y_all)
    print("Mean score:", float(scores.mean()))
    print("Max score:", float(scores.max()))


if __name__ == "__main__":
    main()
