import torch
import torch.nn as nn

import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Callable, Optional, Union
from torch import Tensor


class SelfAttentionDeepSetsEncoder(nn.Module):
    def __init__(
        self, n_features, num_heads=4, hidden_dim=64, output_dim=32, pooling="mean"
    ):
        super().__init__()

        # φ: Per-object MLP
        self.phi = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ρ: Global MLP after pooling
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        assert pooling in ["mean", "cls_token"], (
            "Pooling must be one of: mean, cls_token"
        )

        self.pooling = pooling
        if pooling == "cls_token":
            self.cls_token = nn.Parameter(
                torch.randn(1, 1, hidden_dim)
            )  # Special token used as output representation
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=False
        )

    def forward(self, x):
        x, mask = self.trim(x)  # Remove zero-padded objects
        # Apply φ to each object
        x = self.phi(x)

        if self.pooling == "cls_token":
            # Add cls_token to the beginning of the sequence before permuting
            cls_tokens = self.cls_token.expand(
                x.size(0), -1, -1
            )  # [batch_size, 1, hidden_dim]
            x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, seq_len+1, hidden_dim]
            # Add False mask for cls_token at the start of each sequence
            cls_mask = torch.zeros(
                (mask.size(0), 1), dtype=torch.bool, device=mask.device
            )
            mask = torch.cat((cls_mask, mask), dim=1)  # [batch_size, seq_len+1]

        # SA expects input shape: [seq_len, batch_size, hidden_dim]
        x = x.permute(1, 0, 2)

        # Apply layer normalization before attention
        normed_x = self.norm1(x)

        # Use multi-head attention
        attn_output, _ = self.attention(
            normed_x,
            normed_x,
            normed_x,
            key_padding_mask=mask,  # Use mask to ignore padding
        )

        # Apply pooling
        if self.pooling == "mean":
            mask = ~mask.transpose(
                0, 1
            )  # [seq_len, batch_size] — now True for valid entries
            valid_counts = mask.sum(dim=0).clamp(min=1)

            x = self.norm2(x + attn_output)
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
            x = x.sum(dim=0) / valid_counts.unsqueeze(-1)  # [batch_size, hidden_dim]

        elif self.pooling == "cls_token":
            # Use cls_token as the output representation
            x = self.norm2(x + attn_output)[0]

        # Global MLP
        x = self.rho(x)  # shape: [batch_size, output_dim]
        return x

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


class SelfAttentionDeepSetsFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        n_features=6,
        num_heads=4,
        hidden_dim=64,
        output_dim=32,
        pooling="mean",
    ):
        super().__init__(observation_space, features_dim=output_dim)
        self.self_attention_encoder = SelfAttentionDeepSetsEncoder(
            n_features, num_heads, hidden_dim, output_dim, pooling
        )

    def forward(self, observations):
        return self.self_attention_encoder(observations)


class CustomSelfAttentionDeepSetsPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        n_features=8,
        num_heads=4,
        hidden_dim=64,
        output_dim=32,
        pooling="mean",
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=SelfAttentionDeepSetsFeaturesExtractor,
            features_extractor_kwargs=dict(
                n_features=n_features,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                pooling=pooling,
            ),
            **kwargs,
        )
