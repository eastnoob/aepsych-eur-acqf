"""
Config-driven pipeline integration test for EURAcqfV4.

- Registers EURAcqfV4 so config can reference it
- Loads `experiment_config_v4.ini`
- Runs a 60-trial experiment (15 init + 45 opt) with a virtual user
- Reports unique designs and basic stats
"""

import os
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np

# Add project roots (temp_aepsych first to resolve aepsych)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "temp_aepsych"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "extensions"))

from aepsych.config import Config
from aepsych.strategy import SequentialStrategy

# Register the acquisition in config registry
# Import acquisition function from extensions package path
from dynamic_eur_acquisition.acquisition_function_v4 import EURAcqfV4

Config.register_object(EURAcqfV4)

# Virtual user for rating
from dynamic_eur_acquisition.test.categorical_experiment.virtual_user import (
    VirtualUser,
)


def load_config() -> Config:
    cfg_path = Path(__file__).parent / "experiment_config_v4.ini"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_str = f.read()
    return Config(config_str=cfg_str)


def build_par_specs(cfg: Config):
    parnames = cfg.getlist("common", "parnames", element_type=str)
    specs = []
    for name in parnames:
        ptype = cfg.get(name, "par_type")
        if ptype == "categorical":
            choices = cfg.getlist(name, "choices", element_type=str)
            specs.append({"name": name, "type": "categorical", "choices": choices})
        elif ptype == "integer":
            lb = cfg.getint(name, "lower_bound")
            ub = cfg.getint(name, "upper_bound")
            specs.append({"name": name, "type": "integer", "lb": lb, "ub": ub})
        else:
            # default treat as continuous (not used here)
            lb = float(cfg.get(name, "lower_bound", fallback="0"))
            ub = float(cfg.get(name, "upper_bound", fallback="1"))
            specs.append({"name": name, "type": "continuous", "lb": lb, "ub": ub})
    return parnames, specs


def point_to_design(x: np.ndarray, parnames: List[str], specs: List[Dict]):
    # x is shape (d,) in transformed/original bounds; map to concrete design
    design: Dict = {}
    for i, name in enumerate(parnames):
        spec = specs[i]
        val = x[i]
        if spec["type"] == "categorical":
            idx = int(np.clip(np.round(val), 0, len(spec["choices"]) - 1))
            design[name] = spec["choices"][idx]
        elif spec["type"] == "integer":
            design[name] = int(np.clip(np.round(val), spec["lb"], spec["ub"]))
        else:
            design[name] = float(val)
    return design


def main():
    print("=" * 80)
    print(" EURAcqfV4 Config Pipeline Run")
    print("=" * 80)

    cfg = load_config()
    parnames, specs = build_par_specs(cfg)

    # Strategy
    strat = SequentialStrategy.from_config(cfg)

    # Virtual user
    user = VirtualUser(user_type="balanced", noise_level=0.5, seed=42)

    trial = 0
    designs = []
    ratings = []

    # Run until finished (expected 60 trials)
    while not strat.finished:
        x = strat.gen()
        # x shape: (1, d)
        x_np = x[0].detach().cpu().numpy()
        design = point_to_design(x_np, parnames, specs)
        res = user.rate_design(design)
        rating = res["rating"]

        # record
        designs.append(tuple([design[n] for n in parnames]))
        ratings.append(rating)

        # tell back
        strat.add_data(x, [rating])
        trial += 1
        if trial % 10 == 0:
            print(f"  progress: {trial} trials...")

    # Summary
    unique = len(set(designs))
    print("\nRun complete!")
    print(f"  Total trials: {trial}")
    print(f"  Unique designs: {unique}")
    print(f"  Mean rating: {np.mean(ratings):.4f}")
    print(f"  Max rating: {np.max(ratings):.2f}")


if __name__ == "__main__":
    main()
