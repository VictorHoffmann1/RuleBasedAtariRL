import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


class DeepSetsExtension(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, top_k=64, pooling="mean"):
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
        padding_mask = x.abs().sum(dim=-1) != 0  # [batch_size, num_objects]
        phi_valid_counts = padding_mask.sum(dim=1, keepdim=True)  # [batch_size, 1]
        phi_x = self.phi(x)  # shape: (batch_size, num_objects, hidden_dim)

        # Pairwise interaction
        # Get top-k interactions using attention
        ij_idxs, ij_weights, ij_counts = self.top_k_attention(
            x, padding_mask=padding_mask
        )
        batch_indices = torch.arange(x.size(0), device=x.device).unsqueeze(1).expand(-1, self.top_k)

        # Get (batch, top_k, input_dim) for each of the two interaction ends
        i = ij_idxs[..., 0]
        j = ij_idxs[..., 1]

        x_i = x[batch_indices, i]  # (B, top_k, input_dim)
        x_j = x[batch_indices, j]  # (B, top_k, input_dim)

        object_pairs = torch.cat([x_i, x_j], dim=-1)  # (B, top_k, 2 * input_dim)
        xi_x = self.xi(object_pairs)  # shape: (batch_size, top_k, hidden_dim)
        # Create pairwise mask for valid interactions
        ij_mask = (torch.arange(self.top_k, device=x.device).unsqueeze(0).repeat(x.size(0), 1) < ij_counts.unsqueeze(-1)).unsqueeze(-1) # [batch_size, top_k, 1]

        # Aggregate across objects
        if self.pooling == "mean":
            # Apply padding_mask to phi_x to ignore padded objects
            phi_x = phi_x * padding_mask.unsqueeze(-1)  # [batch_size, num_objects, hidden_dim]
            phi_pooled = phi_x.sum(dim=1) / phi_valid_counts.clamp(min=1)  # Clamp to avoid division by zero

            #xi_x = xi_x * ij_mask  # Apply mask to ignore invalid pairs
            #xi_pooled = xi_x.sum(dim=1) / ij_counts.clamp(min=1)  # Clamp to avoid division by zero
        elif self.pooling == "sum":
            # Apply padding_mask to phi_x to ignore padded objects
            phi_x = phi_x * padding_mask.unsqueeze(-1)  # [batch_size, num_objects, hidden_dim]
            phi_pooled = phi_x.sum(dim=1)

            #xi_x = xi_x * ij_mask  # Apply mask to ignore invalid pairs
            #xi_pooled = xi_x.sum(dim=1)
        elif self.pooling == "max":
            # Apply padding_mask to phi_x to ignore padded objects
            max_padding_mask = ~padding_mask.unsqueeze(-1).expand_as(phi_x) * (
                -1e20
            )  # Set invalid positions to -inf
            phi_x = phi_x + max_padding_mask  # Set invalid positions to -inf
            phi_pooled, _ = phi_x.max(dim=1)

            #max_xi_mask = ~ij_mask * (
            #    -1e20
            #)  # Set invalid positions to -inf  
            #xi_x = xi_x + max_xi_mask  # Set invalid positions to -inf
            #xi_pooled, _ = xi_x.max(dim=1)

        # Do a weighted sum of xi_x using the attention weights
        xi_x = xi_x * ij_weights.unsqueeze(-1)  # [batch_size, top_k, hidden_dim]
        xi_pooled = xi_x.sum(dim=1)

        pooled = torch.cat([phi_pooled, xi_pooled], dim=-1)  # shape: (batch_size, 2 * hidden_dim)

        # Global MLP
        out = self.rho(pooled)  # shape: (batch_size, output_dim)
        return out


class DeepSetsExtensionFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, n_features, hidden_dim, output_dim, top_k, pooling):
        super().__init__(observation_space, features_dim=output_dim)
        self.deepsets_extension_encoder = DeepSetsExtension(n_features, hidden_dim, output_dim, top_k, pooling)

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
        top_k=16,
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
                top_k=top_k,
                pooling=pooling,
            ),
            **kwargs,
        )

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
        self.scale = proj_dim ** 0.5

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
        K = self.key_proj(x)    # (B, L, proj_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, L, L)

        # Build full mask (True = valid, False = invalid)
        full_mask = torch.ones((B, L, L), dtype=torch.bool, device=device)

        if padding_mask is not None:
            padding_mask = padding_mask.bool()
            full_mask &= padding_mask.unsqueeze(2)  # row
            full_mask &= padding_mask.unsqueeze(1)  # column

        # Mask out diagonal (self-attention)
        diag_mask = ~torch.eye(L, dtype=torch.bool, device=device).unsqueeze(0)  # (1, L, L)
        full_mask &= diag_mask

        # Flatten (B, L, L) to (B, L*L)
        attn_scores_flat = attn_scores.view(B, -1)
        full_mask_flat = full_mask.view(B, -1)

        # Set invalid scores to -inf
        attn_scores_flat = attn_scores_flat.masked_fill(~full_mask_flat, float('-inf'))

        # Get top-k across all (valid) entries
        topk_vals, topk_idx = torch.topk(attn_scores_flat, self.top_k, dim=-1, largest=True)
        topk_weights = torch.softmax(topk_vals, dim=-1)  # (B, top_k)

        # Convert flat indices to (i, j)
        row_idx = topk_idx // L
        col_idx = topk_idx % L
        topk_indices = torch.stack([row_idx, col_idx], dim=-1)  # (B, top_k, 2)
        valid_counts = full_mask_flat.sum(dim=-1) # (B,)

        if self.count % 1000 == 0:
            for i in range(5):
                print(f"Top-5 Pairs: {topk_indices[0][i]} with weight {topk_weights[0][i].item()}")
        self.count += 1

        return topk_indices,  topk_weights, valid_counts
