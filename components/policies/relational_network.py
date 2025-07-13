import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


class RelationalNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, top_k=64):
        super().__init__()
        # Self-Interaction MLP
        self.phi = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Pairwise interaction MLP (excluding self-interaction)
        self.xi = nn.Sequential(
            nn.Linear(2 * 6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Top-K attention mechanism to select important interactions
        self.top_k_attention = TopKAttention(
            input_dim=input_dim,  # Input dimension for attention
            proj_dim=hidden_dim,  # Projection dimension for attention
            top_k=top_k,  # Number of top interactions to select
        )
        self.top_k = top_k

        # Global MLP after pooling
        self.rho = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,  # Output dimension of the global MLP
                output_dim,
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
        B, N, D = x.shape  # x: (batch_size, num_objects, input_dim)
        obj_padding_mask = x.abs().sum(dim=-1) != 0  # (B, N)

        # Get top-k interactions using attention
        ij_idxs, ij_weights = self.top_k_attention(
            x, padding_mask=obj_padding_mask
        )  # (B, top_k, 2), (B, top_k)
        self_mask = ij_idxs[..., 0] == ij_idxs[..., 1]  # (B, top_k)

        # Remove from categorical / rgb features from x since they are not meant to be used in pairwise/self-interactions
        # They only exist for the attention head to identify the objects

        x = x[:, :, :6]  # Keep only the first 6 features (e.g., position, speed, size)

        # Batch indices for advanced indexing
        batch_indices = (
            torch.arange(B, device=x.device).unsqueeze(1).expand(-1, self.top_k)
        )  # (B, top_k)

        # Index object vectors
        i_idx = ij_idxs[..., 0]  # (B, top_k)
        j_idx = ij_idxs[..., 1]  # (B, top_k)

        x_i = x[batch_indices, i_idx]  # (B, top_k, D)
        x_j = x[batch_indices, j_idx]  # (B, top_k, D)

        # Self and non-self masks
        self_mask_exp = self_mask.unsqueeze(-1)  # (B, top_k, 1)

        # Prepare self interaction features
        feat_self = self.phi(x_i)  # (B, top_k, H)

        # Prepare pairwise interaction features
        x_pair = torch.cat([x_i, x_j], dim=-1)  # (B, top_k, 2D)
        feat_nonself = self.xi(x_pair)  # (B, top_k, H)

        # Merge features based on mask
        interaction_feat = torch.where(
            self_mask_exp, feat_self, feat_nonself
        )  # (B, top_k, H)

        # Weighted pooling
        ij_weights = ij_weights.unsqueeze(-1)  # (B, top_k, 1)
        pooled_interactions = (interaction_feat * ij_weights).sum(dim=1)  # (B, H)

        # Global MLP
        out = self.rho(pooled_interactions)  # (B, output_dim)
        return out


class RelationalNetworkFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, n_features, hidden_dim, output_dim, top_k):
        super().__init__(observation_space, features_dim=output_dim)
        self.relational_network_encoder = RelationalNetwork(
            n_features, hidden_dim, output_dim, top_k
        )

    def forward(self, observations):
        return self.relational_network_encoder(observations)


class CustomRelationalNetworkPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        n_features=6,
        hidden_dim=64,
        output_dim=32,
        top_k=16,
        **kwargs,
    ):
        kwargs["ortho_init"] = True
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=RelationalNetworkFeaturesExtractor,
            features_extractor_kwargs=dict(
                n_features=n_features,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                top_k=top_k,
            ),
            **kwargs,
        )

    def reinitialize_weights(self):
        """Reinitialize all weights in the policy network"""
        # Reinitialize the relational network encoder
        if hasattr(self.features_extractor, 'relational_network_encoder'):
            self.features_extractor.relational_network_encoder.init_weights()
        
        # Reinitialize all linear layers in the policy
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.orthogonal_(module.weight, gain=torch.sqrt(torch.tensor(2.0)))
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)


class TopKAttention(nn.Module):
    def __init__(self, input_dim, proj_dim, top_k):
        """
        Args:
            input_dim (int): Input feature dimension.
            proj_dim (int): Projection dimension for query/key.
            top_k (int): Number of top attention interactions to return.
        """
        super().__init__()
        self.top_k = top_k
        self.scale = proj_dim**0.5

        self.query_proj = nn.Linear(input_dim, proj_dim)
        self.key_proj = nn.Linear(input_dim, proj_dim)

        self.count = 0

    def forward(self, x, padding_mask=None):
        """
        Args:
            x: Tensor of shape (B, L, D)
            padding_mask: BoolTensor of shape (B, L), True = keep, False = pad
        Returns:
            topk_indices: LongTensor of shape (B, top_k, 2), with (-1, -1) for invalid slots
        """
        B, L, _ = x.shape
        device = x.device

        Q = self.query_proj(x)  # (B, L, proj_dim)
        K = self.key_proj(x)  # (B, L, proj_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, L, L)

        # Build full mask (True = valid, False = invalid)
        full_mask = torch.ones((B, L, L), dtype=torch.bool, device=device)

        if padding_mask is not None:
            padding_mask = padding_mask.bool()
            full_mask &= padding_mask.unsqueeze(2)  # row
            full_mask &= padding_mask.unsqueeze(1)  # column

        # Flatten (B, L, L) to (B, L*L)
        attn_scores_flat = attn_scores.view(B, -1)
        full_mask_flat = full_mask.view(B, -1)

        # Set invalid scores to -inf
        attn_scores_flat = attn_scores_flat.masked_fill(~full_mask_flat, float("-inf"))

        # Get top-k across all (valid) entries
        topk_vals, topk_idx = torch.topk(
            attn_scores_flat, self.top_k, dim=-1, largest=True
        )
        topk_weights = torch.softmax(topk_vals, dim=-1)  # (B, top_k)

        # Convert flat indices to (i, j)
        row_idx = topk_idx // L
        col_idx = topk_idx % L
        topk_indices = torch.stack([row_idx, col_idx], dim=-1)  # (B, top_k, 2)
        # if self.count % 1000 == 0:
        #    for i in range(10):
        #        print(
        #            f"Top-10 Pairs: {topk_indices[0, i, 0].item()} -> {topk_indices[0, i, 1].item()} with weight {topk_weights[0][i].item():.2f}"
        #        )
        # self.count += 1

        return topk_indices, topk_weights
