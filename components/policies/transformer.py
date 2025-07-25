import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


class CustomTransformer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048):
        """Custom Transformer Encoder with no Layer Norm, no Dropout and no Residual connections"""
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0)

        # Feedforward layers
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.activation = F.relu

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # Self-attention block
        src, _ = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
        )
        # Feedforward block
        src = self.linear2(self.activation(self.linear1(src)))
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, n_features, num_heads, num_layers, hidden_dim=32):
        super().__init__()
        self.n_features = n_features
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.embedding = nn.Linear(n_features, hidden_dim)
        encoder_layer = CustomTransformer(d_model=hidden_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, hidden_dim)
        )  # Special token used as output representation

    def forward(self, x):
        mask = x.abs().sum(dim=-1) != 0  # [batch_size, num_objects]
        # Apply embedding layer
        x = self.embedding(x)
        # Transformer expects input shape: [seq_len, batch_size, n_features]
        x = x.permute(1, 0, 2)
        # Add cls_token to the beginning of the sequence
        cls_tokens = self.cls_token.expand(-1, x.size(1), -1)
        x = torch.cat((cls_tokens, x), dim=0)
        # Add False mask for cls_token at the start of each sequence
        cls_mask = torch.zeros((mask.size(0), 1), dtype=torch.bool, device=mask.device)
        mask = torch.cat((cls_mask, mask), dim=1)  # [batch_size, seq_len+1]
        # Pass mask as src_key_padding_mask (expects shape [batch_size, seq_len+1])
        x = self.transformer(x, src_key_padding_mask=mask)
        # Return the cls token output
        return x[0]  # Take the cls_token output (first token)


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self, observation_space, n_features=6, num_heads=4, num_layers=2, hidden_dim=32
    ):
        super().__init__(observation_space, features_dim=hidden_dim)
        self.transformer_encoder = TransformerEncoder(
            n_features, num_heads, num_layers, hidden_dim
        )

    def forward(self, observations):
        return self.transformer_encoder(observations)


class CustomTransformerPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        n_features=8,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=TransformerFeaturesExtractor,
            features_extractor_kwargs=dict(
                n_features=n_features,
                num_heads=num_heads,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
            ),
            **kwargs,
        )
