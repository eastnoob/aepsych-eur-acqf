#!/usr/bin/env python3
"""
HardExclusionGenerator (legacy): moved under legacy subpackage.
For current usage, prefer EURAcqfV4 with pool-based generator that avoids reselection.
"""
from __future__ import annotations

import time
import torch
from aepsych.generators.acqf_grid_search_generator import AcqfGridSearchGenerator
from aepsych.models.base import AEPsychModelMixin
from aepsych.utils_logging import getLogger

logger = getLogger()


class HardExclusionGenerator(AcqfGridSearchGenerator):
    def _gen(
        self,
        num_points: int,
        model: AEPsychModelMixin,
        fixed_features: dict[int, float] | None = None,
        **gen_options,
    ) -> torch.Tensor:
        logger.info("Starting gen...")
        starttime = time.time()

        grid, acqf_vals = self._eval_acqf(
            self.samps, model, fixed_features, **gen_options
        )

        valid_mask = ~torch.isinf(acqf_vals)
        valid_indices = torch.where(valid_mask)[0]

        if len(valid_indices) == 0:
            logger.warning(
                "[HardExclusionGenerator] All candidates have -inf scores! Falling back to random selection."
            )
            random_indices = torch.randperm(len(acqf_vals))[:num_points]
            new_candidate = grid[random_indices]
        elif len(valid_indices) < num_points:
            logger.warning(
                f"[HardExclusionGenerator] Only {len(valid_indices)} valid candidates, but {num_points} requested. Using all valid candidates."
            )
            new_candidate = grid[valid_indices]
        else:
            valid_scores = acqf_vals[valid_indices]
            _, topk_in_valid = torch.topk(valid_scores, num_points)
            selected_indices = valid_indices[topk_in_valid]
            new_candidate = grid[selected_indices]

            logger.info(
                f"[HardExclusionGenerator] Selected {num_points} from {len(valid_indices)} valid candidates (filtered {len(acqf_vals) - len(valid_indices)} -inf)"
            )

        if len(new_candidate.shape) == 3 and new_candidate.shape[1] == 1:
            new_candidate = new_candidate.squeeze(1)

        logger.info(f"Gen done, time={time.time()-starttime}")
        return new_candidate
