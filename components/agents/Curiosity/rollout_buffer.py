from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize

import torch
import numpy as np
from typing import Optional, NamedTuple
from collections.abc import Generator


class CuriosityRolloutBufferSamples(NamedTuple):
    """
    Samples from the rollout buffer for curiosity-driven learning.
    Extends the standard RolloutBufferSamples to include next observations.
    """

    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    next_observations: torch.Tensor


class CuriosityRolloutBuffer(RolloutBuffer):
    """
    Custom rollout buffer for curiosity-driven learning that stores next observations
    in addition to the standard rollout buffer data.
    """

    def __init__(self, *args, **kwargs):
        self.next_observations = None
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        """Reset the buffer."""
        super().reset()
        if self.next_observations is not None:
            self.next_observations = np.zeros_like(self.observations)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        next_obs: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add elements to the buffer.

        Args:
            obs: Observation
            action: Action
            reward: Reward
            episode_start: Start of episode signal
            value: Estimated value of the observation
            log_prob: Log probability of the action
            next_obs: Next observation (optional, will be computed if not provided)
        """
        # Initialize next_observations buffer if needed
        if self.next_observations is None:
            self.next_observations = np.zeros(self.observations.shape, dtype=np.float32)

        # Store next observation
        if next_obs is not None:
            self.next_observations[self.pos] = np.array(next_obs).copy()
        else:
            # If next_obs not provided, we'll compute it later based on the next step's observation
            # For now, just use the current observation as placeholder
            self.next_observations[self.pos] = np.array(obs).copy()

        # Call parent method to handle standard rollout buffer storage
        super().add(obs, action, reward, episode_start, value, log_prob)

    def compute_next_observations(self) -> None:
        """
        Compute next observations by shifting observations by one step.
        This should be called after collecting all rollout data but before training.
        """
        if self.next_observations is None:
            return

        for step in range(self.buffer_size - 1):
            # For each step, next observation is the observation at step + 1
            self.next_observations[step] = self.observations[step + 1]

        # For the last step, handle episode boundaries
        # If episode ends, next_obs should ideally be the reset observation
        # For simplicity, we'll use the same observation as the last step
        self.next_observations[self.buffer_size - 1] = self.observations[
            self.buffer_size - 1
        ]

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[CuriosityRolloutBufferSamples, None, None]:
        """
        Get samples from the buffer.

        Args:
            batch_size: Size of the batches to yield

        Yields:
            Batches of rollout buffer samples including next observations
        """
        assert self.full, "Rollout buffer must be full before sampling"

        # Compute next observations before sampling
        self.compute_next_observations()

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
                "next_observations",
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
    ) -> CuriosityRolloutBufferSamples:
        """
        Get samples from the buffer at the given indices.

        Args:
            batch_inds: Indices of the samples to get
            env: VecNormalize environment (for normalization)

        Returns:
            Batch of rollout buffer samples including next observations
        """
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.next_observations[batch_inds],
        )
        return CuriosityRolloutBufferSamples(*tuple(map(self.to_torch, data)))
