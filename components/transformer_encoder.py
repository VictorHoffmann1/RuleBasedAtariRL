import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


class TransformerEncoder(nn.Module):
    def __init__(self, n_features, num_heads, num_layers):
        super().__init__()
        self.n_features = n_features
        self.num_heads = num_heads
        self.num_layers = num_layers

        encoder_layer = nn.TransformerEncoderLayer(d_model=n_features, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, n_features)
        )  # Special token used as output representation

    def forward(self, x):
        # x shape: [batch_size, seq_len, n_features]
        # Compute mask: True for valid, False for padding (all zeros)
        mask = x.abs().sum(dim=-1) == 0  # [batch_size, seq_len]
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
    def __init__(self, observation_space, n_features, num_heads=2, num_layers=2):
        super().__init__(observation_space, features_dim=n_features)
        self.transformer_encoder = TransformerEncoder(n_features, num_heads, num_layers)

    def forward(self, observations):
        return self.transformer_encoder(observations)


class CustomTransformerPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=TransformerFeaturesExtractor,
            features_extractor_kwargs=dict(n_features=8, num_heads=4, num_layers=2),
            **kwargs,
        )
