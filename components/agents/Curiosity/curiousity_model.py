from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_device


class RepresentationNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, top_k=64):
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
                output_dim,  # Final output dimension
            ),
        )

        self.pairwise_auxiliary_mlp = nn.Linear(hidden_dim, 2)
        self.interaction_feat = None  # Store interaction features for auxiliary loss
        self.x_i, self.x_j = None, None  # Initialize to None for later use
        self.auxiliary_criterion = nn.BCEWithLogitsLoss()

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
        self.x_i = x[batch_indices, i_idx]  # (B, top_k, 6)
        self.x_j = x[batch_indices, j_idx]  # (B, top_k, 6)

        # Parallel feature computation
        feat_self = self.phi(self.x_i)  # (B, top_k, H)
        feat_pair = self.xi(torch.cat([self.x_i, self.x_j], dim=-1))  # (B, top_k, H)

        # Efficient conditional selection
        self.interaction_feat = torch.where(
            self_mask.unsqueeze(-1), feat_self, feat_pair
        )  # (B, top_k, H)

        # Optimized weighted pooling
        pooled = torch.sum(
            self.interaction_feat * ij_weights.unsqueeze(-1), dim=1
        )  # (B, H)

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


class CuriosityModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_space_size):
        super().__init__()
        self.action_space_size = action_space_size

        # Predicts the action that caused transition from prev_state to current_state
        self.action_predictor = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_space_size),
        )

        # Predicts current state from previous state + action
        self.state_predictor = nn.Sequential(
            nn.Linear(input_dim + action_space_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, hidden_state, action, next_hidden_state):
        one_hot_action = F.one_hot(action, num_classes=self.action_space_size).float()
        predicted_action = self.action_predictor(
            torch.cat([hidden_state, next_hidden_state], dim=-1)
        )
        predicted_state = self.state_predictor(
            torch.cat([hidden_state, one_hot_action], dim=-1)
        )

        action_loss = F.cross_entropy(predicted_action, action.long())
        state_loss = torch.mean((predicted_state - next_hidden_state) ** 2, dim=-1)
        return action_loss, state_loss


class FeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, n_features, hidden_dim, output_dim, top_k):
        super().__init__(observation_space, features_dim=output_dim)
        self.encoder = RepresentationNetwork(n_features, hidden_dim, output_dim, top_k)

    def forward(self, observations):
        return self.encoder(observations)


class CuriosityPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        n_features=6,
        hidden_dim=64,
        output_dim=64,
        top_k=16,
        **kwargs,
    ):
        kwargs["ortho_init"] = True
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=FeaturesExtractor,
            features_extractor_kwargs=dict(
                n_features=n_features,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                top_k=top_k,
            ),
            **kwargs,
        )

        self.curiosity_module = CuriosityModule(output_dim, hidden_dim, action_space.n)

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

    def curiosity_loss(self, state, action, next_state):
        """
        Compute the curiosity loss based on the hidden state, action, and next hidden state.
        """
        hidden_state = self.features_extractor(state)
        next_hidden_state = self.features_extractor(next_state)
        return self.curiosity_module(hidden_state, action, next_hidden_state)
