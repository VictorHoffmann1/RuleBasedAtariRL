import numpy as np
import torch
from typing import Optional, Union, NamedTuple
from stable_baselines3.common.buffers import ReplayBuffer
import gymnasium as gym


class BiasedReplayBufferSamples(NamedTuple):
    """
    Named tuple for storing replay buffer samples with bias toward non-zero rewards.
    Contains the current observation, current action, next observation, and reward.
    """

    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    rewards: torch.Tensor


class BiasedReplayBuffer(ReplayBuffer):
    """
    Replay buffer with 33% bias toward non-zero rewards, derived from StableBaselines3 ReplayBuffer.

    This buffer stores transitions as a FIFO queue but maintains separate tracking of
    zero and non-zero reward transitions to enable biased sampling.

    :param buffer_size: Max number of elements in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable memory efficient variant
    :param handle_timeout_termination: Handle timeout termination separately
    :param non_zero_reward_bias: Probability of sampling non-zero reward transitions (default: 0.33)
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        non_zero_reward_bias: float = 0.33,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )

        self.non_zero_reward_bias = non_zero_reward_bias

        # Track indices of zero and non-zero reward transitions
        self.zero_reward_indices = []
        self.non_zero_reward_indices = []

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list,
    ) -> None:
        """
        Add elements to the replay buffer and track reward types.
        """
        # Store the current position before adding
        current_pos = self.pos

        # Call parent's add method to store the transition
        super().add(obs, next_obs, action, reward, done, infos)

        # Track whether this transition has non-zero reward
        # Handle both single environment and vectorized environments
        if isinstance(reward, np.ndarray):
            reward_value = reward.item() if reward.size == 1 else reward[0]
        else:
            reward_value = reward

        # Update indices tracking
        if reward_value != 0:
            # Remove from zero reward indices if it was there
            if current_pos in self.zero_reward_indices:
                self.zero_reward_indices.remove(current_pos)
            # Add to non-zero reward indices if not already there
            if current_pos not in self.non_zero_reward_indices:
                self.non_zero_reward_indices.append(current_pos)
        else:
            # Remove from non-zero reward indices if it was there
            if current_pos in self.non_zero_reward_indices:
                self.non_zero_reward_indices.remove(current_pos)
            # Add to zero reward indices if not already there
            if current_pos not in self.zero_reward_indices:
                self.zero_reward_indices.append(current_pos)

        # Clean up indices that are outside the current buffer range
        # This handles the FIFO nature when buffer wraps around
        valid_range = set(range(min(self.size(), self.buffer_size)))
        self.zero_reward_indices = [
            idx for idx in self.zero_reward_indices if idx in valid_range
        ]
        self.non_zero_reward_indices = [
            idx for idx in self.non_zero_reward_indices if idx in valid_range
        ]

    def sample(
        self, batch_size: int, env: Optional[object] = None
    ) -> BiasedReplayBufferSamples:
        """
        Sample elements from the replay buffer with 33% bias toward non-zero rewards.

        :param batch_size: Number of samples to return
        :param env: Associated VecNormalize object (unused in this implementation)
        :return: BiasedReplayBufferSamples containing observations, actions, next_observations, and rewards
        """
        if self.size() == 0:
            raise ValueError("Cannot sample from empty buffer")

        # Calculate how many samples should be non-zero reward vs zero reward
        n_non_zero_samples = int(batch_size * self.non_zero_reward_bias)
        n_zero_samples = batch_size - n_non_zero_samples

        # Get available indices
        available_non_zero = len(self.non_zero_reward_indices)
        available_zero = len(self.zero_reward_indices)

        # Adjust sampling if we don't have enough of one type
        if available_non_zero < n_non_zero_samples:
            # Not enough non-zero reward samples, take what we have and fill with zero-reward
            n_non_zero_samples = available_non_zero
            n_zero_samples = batch_size - n_non_zero_samples

        if available_zero < n_zero_samples:
            # Not enough zero reward samples, take what we have and fill with non-zero
            n_zero_samples = available_zero
            n_non_zero_samples = batch_size - n_zero_samples

        # Sample indices
        sampled_indices = []

        # Sample non-zero reward indices (always allow replacement to maintain bias)
        if n_non_zero_samples > 0 and available_non_zero > 0:
            non_zero_sample_indices = np.random.choice(
                self.non_zero_reward_indices,
                size=n_non_zero_samples,
                replace=True,  # Always allow replacement to maintain desired bias ratio
            )
            sampled_indices.extend(non_zero_sample_indices)

        # Sample zero reward indices (always allow replacement to maintain bias)
        if n_zero_samples > 0 and available_zero > 0:
            zero_sample_indices = np.random.choice(
                self.zero_reward_indices,
                size=n_zero_samples,
                replace=True,  # Always allow replacement to maintain desired bias ratio
            )
            sampled_indices.extend(zero_sample_indices)

        # If we still don't have enough samples, fill with random samples
        while len(sampled_indices) < batch_size:
            remaining_needed = batch_size - len(sampled_indices)
            additional_indices = np.random.choice(
                self.size(), size=remaining_needed, replace=False
            )
            sampled_indices.extend(additional_indices)

        # Ensure we have exactly batch_size samples
        sampled_indices = sampled_indices[:batch_size]

        # Convert to numpy array and ensure proper data types
        batch_inds = np.array(sampled_indices, dtype=np.int64)

        # Get the actual data using parent class method
        return self._get_samples(batch_inds, env)

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[object] = None
    ) -> BiasedReplayBufferSamples:
        """
        Get samples from the buffer and return only the required attributes.

        :param batch_inds: Indices of samples to retrieve
        :param env: Associated VecNormalize object (unused)
        :return: BiasedReplayBufferSamples with observations, actions, next_observations, and rewards
        """
        # Sample random batch from replay buffer
        # Get observations
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, 0, :], env
            )
        else:
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds, 0, :], env
            )

        # Get current observations, actions, and rewards
        obs = self._normalize_obs(self.observations[batch_inds, 0, :], env)
        actions = self.actions[batch_inds, 0, :]
        rewards = self.rewards[batch_inds, 0]

        # Convert to tensors
        data = (
            obs,
            actions,
            next_obs,
            rewards,
        )

        return BiasedReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def _normalize_obs(
        self, obs: np.ndarray, env: Optional[object] = None
    ) -> np.ndarray:
        """
        Helper to normalize observations if VecNormalize env is provided.

        :param obs: Observations to normalize
        :param env: VecNormalize environment (if any)
        :return: Normalized observations
        """
        if env is not None and hasattr(env, "normalize_obs"):
            return env.normalize_obs(obs)
        return obs

    def reset(self) -> None:
        """
        Reset the buffer and clear index tracking.
        """
        super().reset()
        self.zero_reward_indices = []
        self.non_zero_reward_indices = []

    def get_stats(self) -> dict:
        """
        Get statistics about the current buffer state.

        :return: Dictionary with buffer statistics
        """
        total_size = self.size()
        n_zero_rewards = len(self.zero_reward_indices)
        n_non_zero_rewards = len(self.non_zero_reward_indices)

        return {
            "total_size": total_size,
            "zero_reward_transitions": n_zero_rewards,
            "non_zero_reward_transitions": n_non_zero_rewards,
            "zero_reward_percentage": n_zero_rewards / total_size
            if total_size > 0
            else 0,
            "non_zero_reward_percentage": n_non_zero_rewards / total_size
            if total_size > 0
            else 0,
            "bias_toward_non_zero": self.non_zero_reward_bias,
        }
