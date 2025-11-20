#!/usr/bin/env python3
"""
HardExclusionGenerator: Generator with true hard exclusion logic

This generator extends AcqfGridSearchGenerator to implement true hard exclusion:
1. Evaluate acquisition function on candidates
2. **Filter out all -inf scores before calling torch.topk**
3. Select top-k from remaining candidates

This ensures torch.topk never selects previously sampled designs,
even when k > number of non-inf candidates.
"""
from __future__ import annotations

import time

import torch
from aepsych.generators.acqf_grid_search_generator import AcqfGridSearchGenerator
from aepsych.models.base import AEPsychModelMixin
from aepsych.utils_logging import getLogger

logger = getLogger()


class HardExclusionGenerator(AcqfGridSearchGenerator):
    """
    Generator that implements true hard exclusion by filtering -inf values before topk.

    Key improvement over base class:
    - Filters out candidates with -inf scores before torch.topk
    - This prevents torch.topk from selecting previously sampled designs
    - Works in conjunction with acquisition functions that mark sampled designs as -inf
    """

    def _gen(
        self,
        num_points: int,
        model: AEPsychModelMixin,
        fixed_features: dict[int, float] | None = None,
        **gen_options,
    ) -> torch.Tensor:
        """
        Generates the next query points with hard exclusion.

        Args:
            num_points (int): The number of points to query.
            model (AEPsychModelMixin): The fitted model used to evaluate the acquisition function.
            fixed_features: (dict[int, float], optional): Parameters that are fixed to specific values.
            gen_options (dict): Additional options for generating points.

        Returns:
            torch.Tensor: Next set of points to evaluate, with shape [num_points x dim].
        """
        logger.info("Starting gen...")
        starttime = time.time()

        grid, acqf_vals = self._eval_acqf(
            self.samps, model, fixed_features, **gen_options
        )

        # ✅ Key fix: Filter out -inf values before topk
        valid_mask = ~torch.isinf(acqf_vals)  # True for non-inf values
        valid_indices = torch.where(valid_mask)[0]

        if len(valid_indices) == 0:
            # Extreme case: all candidates are -inf (should never happen with large samps)
            logger.warning(
                "[HardExclusionGenerator] All candidates have -inf scores! Falling back to random selection."
            )
            random_indices = torch.randperm(len(acqf_vals))[:num_points]
            new_candidate = grid[random_indices]
        elif len(valid_indices) < num_points:
            # Warning: fewer valid candidates than requested
            logger.warning(
                f"[HardExclusionGenerator] Only {len(valid_indices)} valid candidates, but {num_points} requested. Using all valid candidates."
            )
            new_candidate = grid[valid_indices]
        else:
            # Normal case: select top-k from valid candidates
            valid_scores = acqf_vals[valid_indices]
            _, topk_in_valid = torch.topk(valid_scores, num_points)
            selected_indices = valid_indices[topk_in_valid]
            new_candidate = grid[selected_indices]

            logger.info(
                f"[HardExclusionGenerator] Selected {num_points} from {len(valid_indices)} valid candidates (filtered {len(acqf_vals) - len(valid_indices)} -inf)"
            )

        # Remove q dimension if present (grid has shape [samps, 1, dim])
        # Result should be [num_points, dim] for proper parameter transforms
        if len(new_candidate.shape) == 3 and new_candidate.shape[1] == 1:
            new_candidate = new_candidate.squeeze(1)

        logger.info(f"Gen done, time={time.time()-starttime}")
        return new_candidate
