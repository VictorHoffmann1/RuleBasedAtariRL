from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize

import torch
import numpy as np
from typing import Optional, NamedTuple
from collections.abc import Generator


class OCZeroRolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    old_actions: torch.Tensor
    rewards: torch.Tensor


class OCZeroRolloutBuffer(RolloutBuffer):
    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[OCZeroRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "rewards",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> OCZeroRolloutBufferSamples:
        # Get old actions and rewards from the previous timestep
        step_indices = batch_inds // self.n_envs
        env_indices = batch_inds % self.n_envs
        old_step_indices = np.maximum(step_indices - 1, 0)
        old_batch_inds = old_step_indices * self.n_envs + env_indices

        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.actions[old_batch_inds],
            self.rewards[batch_inds].flatten(),
        )
        return OCZeroRolloutBufferSamples(*tuple(map(self.to_torch, data)))
