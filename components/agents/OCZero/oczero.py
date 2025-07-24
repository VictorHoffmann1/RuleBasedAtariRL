from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_device


class RepresentationNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, top_k=64):
        super().__init__()
        # Self-Interaction MLP
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Pairwise interaction MLP (excluding self-interaction)
        self.xi = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Top-K attention mechanism to select important interactions
        self.top_k_attention = TopKAttention(
            input_dim=input_dim,  # Input dimension for attention
            proj_dim=hidden_dim,  # Projection dimension for attention
            top_k=top_k,  # Number of top interactions to select
        )

        # Global MLP after pooling
        self.rho = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                hidden_dim,  # Output dimension of the global MLP
                hidden_dim,
            ),
        )

        self.hidden_dim = hidden_dim

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=torch.sqrt(torch.tensor(2.0)))
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x, obj_padding_mask = self.trim(x)  # Remove zero-padded objects
        B = x.shape[0]

        # Get top-k interactions
        ij_idxs, ij_weights = self.top_k_attention(x, padding_mask=obj_padding_mask)

        top_k = ij_idxs.shape[1]  # Number of top-k interactions

        # Efficient self-mask computation
        self_mask = ij_idxs[..., 0] == ij_idxs[..., 1]  # (B, top_k)
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, top_k)

        # Vectorized indexing
        i_idx, j_idx = ij_idxs[..., 0], ij_idxs[..., 1]
        x_i = x[batch_indices, i_idx]  # (B, top_k, 6)
        x_j = x[batch_indices, j_idx]  # (B, top_k, 6)

        # Parallel feature computation
        feat_self = self.phi(x_i)  # (B, top_k, H)
        feat_pair = self.xi(torch.cat([x_i, x_j], dim=-1))  # (B, top_k, H)

        # Efficient conditional selection
        interaction_feat = torch.where(
            self_mask.unsqueeze(-1), feat_self, feat_pair
        )  # (B, top_k, H)

        # Optimized weighted pooling
        pooled = torch.sum(interaction_feat * ij_weights.unsqueeze(-1), dim=1)  # (B, H)

        return self.rho(pooled)

    @staticmethod
    def trim(x):
        """
        Remove trailing zero-padded objects from the input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, D) with zero-padded objects.
        Returns:
            torch.Tensor: Tensor with zero-padded objects trimmed, shape (B, max_valid_N, D).
        """

        obj_padding_mask = x.abs().sum(dim=-1) != 0  # (B, N)

        max_valid = obj_padding_mask.sum(dim=1).max()

        return x[:, :max_valid, :], obj_padding_mask[
            :, :max_valid
        ]  # Return trimmed tensor and mask

    @staticmethod
    def is_collision(x, y, eps=1.0):
        """
        Check if two objects collide based on their positions, shapes and speeds.
        Args:
            x (torch.Tensor): Position tensor of shape (B, top_k, D).
            y (torch.Tensor): Position tensor of shape (B, top_k, D).
        Returns:
            torch.Tensor: Boolean tensor indicating collisions, shape (B, top_k).
        """
        # Get positions, shapes and speeds
        x_pos, x_speed, x_shape = x[..., :2], x[..., 2:4], x[..., 4:6]
        y_pos, y_speed, y_shape = y[..., :2], y[..., 2:4], y[..., 4:6]

        # Check if the objects collide / overlap
        cond1 = torch.abs(x_pos - y_pos) < (x_shape + y_shape) / 2

        # Check if there will be a collision in the next step, since there can be object detection errors
        # when two objects touch, they can be detected as one single object
        cond2 = (
            torch.abs(x_pos + x_speed - y_pos - y_speed) < (x_shape + y_shape) / 2 + eps
        )

        return (cond1[..., 0] & cond1[..., 1]) | (cond2[..., 0] & cond2[..., 1])


class TopKAttention(nn.Module):
    """
    Top-K attention mechanism to select important interactions.
    This module computes pairwise attention scores and selects the top-k pairs.
    """

    def __init__(self, input_dim, proj_dim, top_k, verbose=False):
        """
        Initialize the Top-K attention mechanism.
        """
        super().__init__()
        self.top_k = top_k
        self.scale = proj_dim**-0.5

        self.query_proj = nn.Linear(input_dim, proj_dim)  # Query projection
        self.key_proj = nn.Linear(input_dim, proj_dim)  # Key projection

        self.verbose = verbose
        if verbose:
            self.count = 0

    def forward(self, x, padding_mask=None):
        B, L, _ = x.shape

        Q = self.query_proj(x)  # (B, L, proj_dim)
        K = self.key_proj(x)  # (B, L, proj_dim)

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Efficient masking
        if padding_mask is not None:
            # Create 2D mask efficiently
            mask_2d = padding_mask.unsqueeze(2) & padding_mask.unsqueeze(1)
            attn_scores.masked_fill_(~mask_2d, float("-inf"))

        # Flatten and get top-k
        attn_scores_flat = attn_scores.view(B, -1)
        topk_vals, topk_idx = torch.topk(
            attn_scores_flat, min(self.top_k, L**2), dim=-1, largest=True
        )
        topk_weights = F.softmax(topk_vals, dim=-1)

        # Convert indices efficiently
        row_idx = topk_idx // L
        col_idx = topk_idx % L
        topk_indices = torch.stack([row_idx, col_idx], dim=-1)

        if self.verbose:
            if self.count % 1000 == 0:  # Print every 1000 calls
                print(f"Top-{min(L**2, 10)} Pairs:")
                for i in range(min(L**2, 10)):
                    print(
                        f"{topk_indices[0, i, 0].item()} -> {topk_indices[0, i, 1].item()} with weight {topk_weights[0][i].item():.2f}"
                    )
                print(f"Number of objects: {L}")
            self.count += 1

        return topk_indices, topk_weights


# TODO: Maybe do a Dynamics Network specific for the pairwise interactions MLP


class DynamicsNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        proj_dim,
        action_space_size,
        reward_clipped=True,
        reward_scale=10.0,
        lambd=0.0051,
    ):
        """Dynamics network: Predicts dynamics of the environment to aid learning.

        The network learns three auxiliary tasks:
        1. Action Prediction: Given previous and current states, predict the action that caused the transition
        2. State Prediction: Given previous state and action, predict the current state
        3. Reward Prediction: Given the predicted current state, predict the reward received

        Parameters
        ----------
        input_dim: int
            dimensionality of the hidden state representations
        hidden_dim: int
            hidden layer size for the prediction networks
        proj_dim: int
            projection dimension for contrastive learning
        action_space_size: int
            number of actions in the environment
        reward_clipped: bool
            whether rewards are clipped (use Tanh activation)
        reward_scale: float
            scaling factor for reward prediction loss
        lambd: float
            weight for off-diagonal terms in Barlow Twins loss
        """
        super().__init__()

        self.action_embedding = nn.Linear(action_space_size, input_dim)
        self.action_space_size = action_space_size

        # Predicts the action that caused transition from prev_state to current_state
        self.action_predictor = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_space_size),
        )

        # Predicts current state from previous state + action
        self.state_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )
        self.prev_hidden_state = None

        # Predicts reward from the predicted current state
        self.reward_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Tanh() if reward_clipped else nn.Identity(),
        )
        self.reward_scale = reward_scale

        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
        self.batch_norm = nn.BatchNorm1d(proj_dim, affine=False)
        self.lambd = lambd

    def forward(self, hidden_state, prev_action, reward):
        if self.prev_hidden_state is None:
            self.prev_hidden_state = hidden_state.detach()
            return (
                torch.tensor(0.0, device=hidden_state.device),
                torch.tensor(0.0, device=hidden_state.device),
                torch.tensor(0.0, device=hidden_state.device),
            )

        prev_action = prev_action.long().flatten()

        # Predict the action that led FROM previous state TO current state
        predicted_action = self.action_predictor(
            torch.cat([hidden_state, self.prev_hidden_state], dim=-1)
        )

        # Current State Prediction: predict current state from previous state + action
        # Add action embedding to the previous hidden states
        one_hot_action = F.one_hot(
            prev_action, num_classes=self.action_space_size
        ).float()
        x = self.prev_hidden_state + self.action_embedding(
            one_hot_action
        )  # (B, input_dim)

        # Predict current hidden state based on previous hidden state and action that was taken
        predicted_current_state = self.state_predictor(x)  # (B, input_dim)

        # Reward prediction: predict reward received when transitioning to current state
        predicted_reward = self.reward_predictor(predicted_current_state).flatten()

        # Compute losses
        # Action prediction: predict which action led to current state
        action_prediction_loss = F.cross_entropy(predicted_action, prev_action)
        # State prediction: predict current state from previous state + action
        projection_loss = self.project(predicted_current_state, hidden_state)
        # Reward prediction: predict reward received for this transition
        reward_prediction_loss = F.mse_loss(
            self.reward_scale * predicted_reward, self.reward_scale * reward
        )

        # Update previous hidden state (detach to avoid gradient computation issues)
        self.prev_hidden_state = hidden_state.detach()

        return projection_loss, action_prediction_loss, reward_prediction_loss

    def project(self, hidden_state, prev_hidden_state):
        """Project hidden states to a higher space and computes cosine similarity."""
        x1 = self.projector(hidden_state).detach()  # (B, output_dim)
        x2 = self.predictor(self.projector(prev_hidden_state))  # (B, output_dim)

        batch_loss = self.consist_loss_func(x1, x2)
        return batch_loss.mean()

    @staticmethod
    def consist_loss_func(f1, f2):
        """Consistency loss function: similarity loss
        Parameters
        """
        f1 = F.normalize(f1, p=2.0, dim=-1, eps=1e-5)
        f2 = F.normalize(f2, p=2.0, dim=-1, eps=1e-5)
        return -(f1 * f2).sum(dim=1)


class OCZeroFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, n_features, hidden_dim, top_k):
        super().__init__(observation_space, features_dim=hidden_dim)
        self.relational_network_encoder = RepresentationNetwork(
            n_features, hidden_dim, top_k
        )

    def forward(self, observations):
        return self.relational_network_encoder(observations)


class OCZeroMlpExtractor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        projector_dim: int,
        action_space: int,
        net_arch: Union[list[int], dict[str, list[int]]],
        activation_fn: type[nn.Module],
        device: Union[torch.device, str] = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)

        # --- Dynamics Network ---
        self.dynamics_network = DynamicsNetwork(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            proj_dim=projector_dim,
            action_space_size=action_space,
        ).to(device)

        # --- Actor/Critic Network ---
        policy_net: list[nn.Module] = []
        value_net: list[nn.Module] = []
        last_layer_dim_pi = feature_dim
        last_layer_dim_vf = feature_dim

        if isinstance(net_arch, dict):
            # Note: if key is not specified, assume linear network
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
        else:
            pi_layers_dims = vf_layers_dims = net_arch
        # Iterate through the policy layers and build the policy net
        for curr_layer_dim in pi_layers_dims:
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim
        # Iterate through the value layers and build the value net
        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)

    def forward_dynamics(
        self,
        features: torch.Tensor,
        last_action: torch.Tensor,
        reward: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the dynamics network.
        :return: projection_loss, action_prediction_loss, reward_prediction_loss
        """
        return self.dynamics_network(features, last_action, reward)


class OCZeroPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        n_features=6,
        hidden_dim=64,
        projector_dim=512,
        top_k=16,
        **kwargs,
    ):
        self.hidden_dim = hidden_dim
        self.projector_dim = projector_dim

        kwargs["ortho_init"] = True
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=OCZeroFeaturesExtractor,
            features_extractor_kwargs=dict(
                n_features=n_features,
                hidden_dim=hidden_dim,
                top_k=top_k,
            ),
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = OCZeroMlpExtractor(
            self.features_dim,
            self.hidden_dim,
            self.projector_dim,
            self.action_space.n,
            self.net_arch,
            self.activation_fn,
            device=self.device,
        )

    def evaluate_dynamics(
        self,
        observations: torch.Tensor,
        last_actions: torch.Tensor,
        rewards: torch.Tensor,
    ):
        features = self.features_extractor(observations)
        return self.mlp_extractor.forward_dynamics(features, last_actions, rewards)

    def reinitialize_weights(self):
        """Reinitialize all weights in the policy network"""
        # Reinitialize the relational network encoder
        if hasattr(self.features_extractor, "relational_network_encoder"):
            self.features_extractor.relational_network_encoder.init_weights()

        # Reinitialize all linear layers in the policy
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.orthogonal_(
                    module.weight, gain=torch.sqrt(torch.tensor(2.0))
                )
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
