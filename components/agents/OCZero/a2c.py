from typing import Any, Optional, Union

from stable_baselines3.a2c import A2C
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from gymnasium import spaces
from components.agents.OCZero.oczero import OCZeroPolicy
from components.agents.OCZero.rollout_buffer import OCZeroRolloutBuffer
import torch
import torch.nn.functional as F


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
        proj_coef: float = 1.0,
        action_pred_coef: float = 1.0,
        reward_pred_coef: float = 1.0,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
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
            OCZeroRolloutBuffer,
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
        self.action_pred_coef = action_pred_coef
        self.reward_pred_coef = reward_pred_coef

    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):
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

            # Compute auxiliary dynamics losses
            projection_loss, action_prediction_loss, reward_prediction_loss = (
                self.policy.evaluate_dynamics(
                    rollout_data.observations,
                    rollout_data.old_actions,
                    rollout_data.rewards,
                )
            )

            # Combine all losses
            loss = (
                policy_loss
                + self.ent_coef * entropy_loss
                + self.vf_coef * value_loss
                + self.proj_coef * projection_loss
                + self.action_pred_coef * action_prediction_loss
                + self.reward_pred_coef * reward_prediction_loss
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
        self.logger.record(
            "train/action_prediction_loss", action_prediction_loss.item()
        )
        self.logger.record(
            "train/reward_prediction_loss", reward_prediction_loss.item()
        )
        if hasattr(self.policy, "log_std"):
            self.logger.record(
                "train/std", torch.exp(self.policy.log_std).mean().item()
            )
