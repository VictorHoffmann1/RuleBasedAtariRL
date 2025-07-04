import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


class TransformerEncoder(nn.Module):
    def __init__(self, n_features, num_heads, num_layers, hidden_dim=32):
        super().__init__()
        self.n_features = n_features
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.embedding = nn.Linear(n_features, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, n_features)
        )  # Special token used as output representation

    def forward(self, x):
        # x shape: [batch_size, seq_len, n_features]
        # Compute mask: True for valid, False for padding (all zeros)
        mask = x.abs().sum(dim=-1) == 0  # [batch_size, seq_len]
        # Apply embedding layer
        x = self.embedding(x)
        # Transformer expects input shape: [seq_len, batch_size, n_features]
        x = x.permute(1, 0, 2)
        # Add cls_token to the beginning of the sequence
        cls_tokens = self.embedding(self.cls_token).expand(-1, x.size(1), -1)
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
        self, observation_space, n_features, num_heads=4, num_layers=2, hidden_dim=32
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
        num_heads=1,
        num_layers=1,
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
