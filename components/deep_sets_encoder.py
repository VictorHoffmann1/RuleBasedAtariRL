import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


class DeepSets(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, pooling="mean"):
        super().__init__()
        # φ: Per-object MLP
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ρ: Global MLP after pooling
        self.rho = nn.Sequential(
            nn.Linear(
                2 * hidden_dim
                if pooling == "max+mean"
                else hidden_dim,  # Concatenate max and mean pooled features
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,  # Output dimension of the global MLP
                output_dim,
            ),
        )

        if pooling == "learned":
            # Learned pooling layer
            self.pooling_layer = nn.Linear(input_dim, 1)

        assert pooling in ["mean", "sum", "max", "max+mean", "learned"], (
            "Pooling must be one of: mean, sum, max"
        )
        self.pooling = pooling

    def forward(self, x):
        # x: (batch_size, num_objects, input_dim)
        mask = x.abs().sum(dim=-1) != 0  # [batch_size, num_objects]
        valid_counts = mask.sum(dim=1, keepdim=True)  # [batch_size, 1]

        phi_x = self.phi(x)  # shape: (batch_size, num_objects, hidden_dim)

        # Aggregate across objects
        if self.pooling == "mean":
            # Apply mask to phi_x to ignore padded objects
            phi_x = phi_x * mask.unsqueeze(-1)  # [batch_size, num_objects, hidden_dim]
            pooled = phi_x.sum(dim=1) / valid_counts.clamp(
                min=1
            )  # Clamp to avoid division by zero
        elif self.pooling == "sum":
            # Apply mask to phi_x to ignore padded objects
            phi_x = phi_x * mask.unsqueeze(-1)  # [batch_size, num_objects, hidden_dim]
            pooled = phi_x.sum(dim=1)
        elif self.pooling == "max":
            # Apply mask to phi_x to ignore padded objects
            max_mask = ~mask.unsqueeze(-1).expand_as(phi_x) * (
                -1e20
            )  # Set invalid positions to -inf
            phi_x = phi_x + max_mask  # Set invalid positions to -inf
            pooled, _ = phi_x.max(dim=1)
        elif self.pooling == "learned":
            # Learned pooling
            weights = F.softmax(
                self.pooling_layer(x) * mask.float().unsqueeze(-1), dim=1
            )  # [batch_size, num_objects, 1]
            pooled = (weights * phi_x).sum(dim=1)  # [batch_size, hidden_dim]
        elif self.pooling == "max+mean":
            # Max Pooling
            max_mask = ~mask.unsqueeze(-1).expand_as(phi_x) * (-1e20)
            phi_x_max_pool = phi_x + max_mask
            pooled_max, _ = phi_x_max_pool.max(dim=1)

            # Mean Pooling
            phi_x_mean_pool = phi_x * mask.unsqueeze(-1)
            pooled_mean = phi_x_mean_pool.sum(dim=1) / valid_counts.clamp(min=1)

            # Concatenate max and mean pooled features
            pooled = torch.cat((pooled_max, pooled_mean), dim=-1)

        # Global MLP
        out = self.rho(pooled)  # shape: (batch_size, output_dim)
        return out


class DeepSetsFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, n_features, hidden_dim, output_dim, pooling):
        super().__init__(observation_space, features_dim=output_dim)
        self.deepsets_encoder = DeepSets(n_features, hidden_dim, output_dim, pooling)

    def forward(self, observations):
        return self.deepsets_encoder(observations)


class CustomDeepSetPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=DeepSetsFeaturesExtractor,
            features_extractor_kwargs=dict(
                n_features=8, hidden_dim=64, output_dim=32, pooling="max+mean"
            ),
            **kwargs,
        )
