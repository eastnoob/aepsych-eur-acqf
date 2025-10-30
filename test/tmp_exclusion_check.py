import sys, os, numpy as np, torch
from pathlib import Path

root = r"d:\WORKSPACE\python\aepsych-source"
sys.path.insert(0, os.path.join(root, "extensions"))
sys.path.insert(0, os.path.join(root, "temp_aepsych"))

from aepsych.config import Config
from aepsych.transforms.parameters import ParameterTransforms, ParameterTransformedModel
from dynamic_eur_acquisition.acquisition_function_v3 import HardExclusionAcqf

cfg_path = os.path.join(
    root,
    "extensions",
    "dynamic_eur_acquisition",
    "test",
    "categorical_experiment",
    "experiment_config_v3a.ini",
)
config_str = open(cfg_path, "r", encoding="utf-8").read()
config = Config(config_str=config_str)

transforms = ParameterTransforms.from_config(config)


class DummyModel:
    def __init__(self):
        self._train_inputs = None
        self._train_targets = None
        self.stimuli_per_trial = 1
        self._p = torch.nn.Parameter(torch.tensor(0.0))

    def parameters(self):
        return [self._p]

    @property
    def device(self):
        return torch.device("cpu")

    @property
    def train_inputs(self):
        return (self._train_inputs,) if self._train_inputs is not None else None

    @train_inputs.setter
    def train_inputs(self, ti):
        self._train_inputs = ti[0]

    @property
    def train_targets(self):
        return self._train_targets

    @train_targets.setter
    def train_targets(self, tt):
        self._train_targets = tt


base = DummyModel()
model = ParameterTransformedModel(model=base, transforms=transforms)

hist_point = np.array([1.0, 0.0, 22.0, 2.0], dtype=float)
base.train_inputs = (torch.tensor(hist_point, dtype=torch.float64).unsqueeze(0),)
base.train_targets = torch.tensor([8.0], dtype=torch.float64)

acqf = HardExclusionAcqf(model=model)

cands = np.array(
    [
        [1.0000000, 0.0000000, 22.0, 2.0000000],  # exact duplicate
        [1.01, -0.02, 21.6, 2.49],  # should canonicalize to [1,0,22,2]
        [0.49, 0.49, 12.4, 0.51],  # different point
        [2.49, 3.49, 18.49, 0.51],  # different point
    ],
    dtype=float,
)

scores = acqf(torch.tensor(cands, dtype=torch.float64))
print("scores:", scores.tolist())
penalized = sum(1 for v in scores.tolist() if v <= -1e9)
print("penalized:", penalized)
