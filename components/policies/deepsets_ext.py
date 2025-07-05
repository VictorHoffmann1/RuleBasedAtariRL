import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


class DeepSetsExtension(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, pooling="mean"):
        super().__init__()
        # Per-object MLP
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Pairwise interaction MLP
        self.xi = nn.Sequential(
            nn.Linear(2*input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Global MLP after pooling
        self.rho = nn.Sequential(
            nn.Linear(
                2 * hidden_dim,  # Concatenate object and interaction features
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,  # Output dimension of the global MLP
                output_dim,
            ),
        )

        assert pooling in ["mean", "sum", "max"], (
            "Pooling must be one of: mean, sum, max"
        )
        self.pooling = pooling

    def forward(self, x):

        # Per object MLP
        # x: (batch_size, num_objects, input_dim)
        phi_mask = x.abs().sum(dim=-1) != 0  # [batch_size, num_objects]
        phi_valid_counts = phi_mask.sum(dim=1, keepdim=True)  # [batch_size, 1]
        phi_x = self.phi(x)  # shape: (batch_size, num_objects, hidden_dim)

        # Pairwise interaction
        xi_x = self.xi(self.get_pairwise_concatenation(x))  # shape: (batch_size, num_objects^2, hidden_dim)
        pairwise_mask = self.create_pairwise_mask(x)  # shape: (batch_size, num_objects^2, 1)
        xi_valid_counts = pairwise_mask.sum(dim=1)  # [batch_size, 1]



        # Aggregate across objects
        if self.pooling == "mean":
            # Apply phi_mask to phi_x to ignore padded objects
            phi_x = phi_x * phi_mask.unsqueeze(-1)  # [batch_size, num_objects, hidden_dim]
            phi_pooled = phi_x.sum(dim=1) / phi_valid_counts.clamp(min=1)  # Clamp to avoid division by zero

            xi_x = xi_x * pairwise_mask  # Apply pairwise mask
            xi_pooled = xi_x.sum(dim=1) / xi_valid_counts.clamp(min=1)  # Clamp to avoid division by zero
        elif self.pooling == "sum":
            # Apply phi_mask to phi_x to ignore padded objects
            phi_x = phi_x * phi_mask.unsqueeze(-1)  # [batch_size, num_objects, hidden_dim]
            phi_pooled = phi_x.sum(dim=1)

            xi_x = xi_x * pairwise_mask  # Apply pairwise mask
            xi_pooled = xi_x.sum(dim=1)
        elif self.pooling == "max":
            # Apply phi_mask to phi_x to ignore padded objects
            max_phi_mask = ~phi_mask.unsqueeze(-1).expand_as(phi_x) * (
                -1e20
            )  # Set invalid positions to -inf
            phi_x = phi_x + max_phi_mask  # Set invalid positions to -inf
            phi_pooled, _ = phi_x.max(dim=1)

            max_xi_mask = ~pairwise_mask.expand_as(xi_x) * (
                -1e20
            )  # Set invalid positions to -inf
            xi_x = xi_x + max_xi_mask  # Set invalid positions to -inf
            xi_pooled, _ = xi_x.max(dim=1)

        pooled = torch.cat([phi_pooled, xi_pooled], dim=-1)  # shape: (batch_size, 2 * hidden_dim)


        # Global MLP
        out = self.rho(pooled)  # shape: (batch_size, output_dim)
        return out
    
    @staticmethod
    def get_pairwise_concatenation(x):
        batch_size, num_objects, n_input = x.shape

        # Expand and repeat to form all pairwise combinations
        x_i = x.unsqueeze(2).expand(-1, num_objects, num_objects, -1)  # shape: (B, N, N, F)
        x_j = x.unsqueeze(1).expand(-1, num_objects, num_objects, -1)  # shape: (B, N, N, F)

        # Concatenate along the last dimension
        pairwise = torch.cat([x_i, x_j], dim=-1)  # shape: (B, N, N, 2F)

        # Reshape to (B, N², 2F)
        pairwise = pairwise.view(batch_size, num_objects ** 2, 2 * n_input)
        return pairwise
    
    @staticmethod
    def create_pairwise_mask(x):
        """
        Create a mask for pairs (i, j) where the speed of i is not null 
        """
        batch_size, num_objects, n_input = x.shape

        # f_i is the first element in the pair (i, j)
        f_i = x.unsqueeze(2).expand(-1, num_objects, num_objects, -1)  # shape: (B, N, N, F)

        # Extract speed (slice [2:4])
        condition = (f_i[..., 2:4].norm(dim=-1) > 0)  # shape: (B, N, N)

        # Flatten to (B, N², 1)
        mask = condition.view(batch_size, num_objects ** 2, 1)
        return mask


class DeepSetsExtensionFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, n_features, hidden_dim, output_dim, pooling):
        super().__init__(observation_space, features_dim=output_dim)
        self.deepsets_extension_encoder = DeepSetsExtension(n_features, hidden_dim, output_dim, pooling)

    def forward(self, observations):
        return self.deepsets_extension_encoder(observations)


class CustomDeepSetExtensionPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        n_features=6,
        hidden_dim=64,
        output_dim=32,
        pooling="max",
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=DeepSetsExtensionFeaturesExtractor,
            features_extractor_kwargs=dict(
                n_features=n_features,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                pooling=pooling,
            ),
            **kwargs,
        )
