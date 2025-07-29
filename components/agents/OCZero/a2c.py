from typing import Any, Optional, Union

from stable_baselines3.a2c import A2C
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from gymnasium import spaces
from components.agents.OCZero.oczero import OCZeroPolicy
from stable_baselines3.common.buffers import RolloutBuffer
from components.agents.OCZero.replay_buffer import BiasedReplayBuffer
import torch
import torch.nn.functional as F
import numpy as np


class OCZeroA2C(A2C):
    def __init__(
        self,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        n_aux_epochs: int = 20,
        aux_batch_size: int = 32,
        proj_coef: float = 1.0,
        reward_pred_coef: float = 1.0,
        collision_coef: float = 1.0,
        closer_coef: float = 1.0,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        normalize_advantage: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            OCZeroPolicy,
            env,
            learning_rate,
            n_steps,
            gamma,
            gae_lambda,
            ent_coef,
            vf_coef,
            max_grad_norm,
            rms_prop_eps,
            use_rms_prop,
            use_sde,
            sde_sample_freq,
            rollout_buffer_class,
            rollout_buffer_kwargs,
            normalize_advantage,
            stats_window_size,
            tensorboard_log,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model=_init_setup_model,
        )

        self.proj_coef = proj_coef
        self.collision_coef = collision_coef
        self.reward_pred_coef = reward_pred_coef
        self.closer_coef = closer_coef

        self.n_aux_epochs = n_aux_epochs
        self.aux_batch_size = aux_batch_size
        self.replay_buffer = BiasedReplayBuffer(
            buffer_size=2000,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            non_zero_reward_bias=0.33,
        )

    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Transfer rollout buffer data to replay buffer
        self._transfer_rollout_to_replay()

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        projection_losses, reward_prediction_losses, collision_losses, closer_losses = (
            [],
            [],
            [],
            [],
        )

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            # Auxiliary tasks
            for epoch in range(self.n_aux_epochs):
                # Compute auxiliary dynamics losses
                # Only train if we have enough samples in the replay buffer
                if self.replay_buffer.size() >= self.aux_batch_size:
                    replay_data = self.replay_buffer.sample(self.aux_batch_size)

                    if isinstance(self.action_space, spaces.Discrete):
                        # Convert discrete action from float to long
                        actions = replay_data.actions.long().flatten()

                    projection_loss, reward_prediction_loss = (
                        self.policy.evaluate_dynamics(
                            replay_data.observations,
                            actions,
                            replay_data.rewards,
                            replay_data.next_observations,
                        )
                    )

                    collision_loss, closer_loss = (
                        self.policy.features_extractor.encoder.pairwise_auxiliary_loss()
                    )

                    aux_loss = (
                        self.reward_pred_coef * reward_prediction_loss
                        + self.proj_coef * projection_loss
                        + self.collision_coef * collision_loss
                        + self.closer_coef * closer_loss
                    )

                    projection_losses.append(projection_loss.item())
                    reward_prediction_losses.append(reward_prediction_loss.item())
                    collision_losses.append(collision_loss.item())
                    closer_losses.append(closer_loss.item())

                    self.policy.auxiliary_optimizer.zero_grad()
                    aux_loss.backward()
                    self.policy.auxiliary_optimizer.step()

            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()

            values, log_prob, entropy = self.policy.evaluate_actions(
                rollout_data.observations, actions
            )
            values = values.flatten()

            # Normalize advantage (not present in the original implementation)
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

            # Policy gradient loss
            policy_loss = -(advantages * log_prob).mean()

            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns, values)

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -torch.mean(-log_prob)
            else:
                entropy_loss = -torch.mean(entropy)

            # Combine all losses
            loss = (
                policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
            )

            # Optimization step
            self.policy.optimizer.zero_grad()
            # Backpropagate the loss
            loss.backward()

            # Clip grad norm
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        self.logger.record("train/projection_loss", projection_loss.item())
        self.logger.record("train/dynamics_proj_loss", np.mean(projection_losses))
        self.logger.record(
            "train/dynamics_reward_pred_loss", np.mean(reward_prediction_losses)
        )
        self.logger.record("train/dynamics_collision_loss", np.mean(collision_losses))
        self.logger.record("train/dynamics_closer_loss", np.mean(closer_losses))
        if hasattr(self.policy, "log_std"):
            self.logger.record(
                "train/std", torch.exp(self.policy.log_std).mean().item()
            )

    def _transfer_rollout_to_replay(self) -> None:
        """
        Transfer data from rollout buffer to replay buffer for auxiliary training.
        """
        # Convert rollout buffer to numpy arrays for easier manipulation
        buffer_size = self.rollout_buffer.buffer_size
        n_envs = self.rollout_buffer.n_envs

        # Get data from rollout buffer
        observations = (
            self.rollout_buffer.observations
        )  # Shape: (buffer_size, n_envs, *obs_shape)
        actions = (
            self.rollout_buffer.actions
        )  # Shape: (buffer_size, n_envs, *action_shape)
        rewards = self.rollout_buffer.rewards  # Shape: (buffer_size, n_envs)

        # For next observations, we need to shift observations by 1 step
        # Use the next observation at each timestep
        for step in range(buffer_size):
            for env_idx in range(n_envs):
                # Current observation
                obs = observations[step, env_idx : env_idx + 1]  # Keep batch dimension

                # Action taken
                action = actions[step, env_idx : env_idx + 1]  # Keep batch dimension

                # Reward received
                reward = rewards[step, env_idx : env_idx + 1]  # Keep batch dimension

                # Next observation (use next step's observation, or current if last step)
                if step < buffer_size - 1:
                    next_obs = observations[step + 1, env_idx : env_idx + 1]
                else:
                    # For the last step, use the same observation as next_obs
                    # In practice, this should be handled by episode termination
                    next_obs = observations[step, env_idx : env_idx + 1]

                # Done flag (assume not done for simplicity, this could be improved)
                done = np.array([False])

                # Add to replay buffer
                self.replay_buffer.add(
                    obs=obs,
                    next_obs=next_obs,
                    action=action,
                    reward=reward,
                    done=done,
                    infos=[{}],
                )
