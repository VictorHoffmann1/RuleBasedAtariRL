import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


class LSTM(nn.Module):
    def __init__(self, n_features, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_dim, batch_first=True
        )

    def forward(self, x):
        x, mask = self.trim(x)  # Remove zero-padded objects
        _, (h_n, _) = self.lstm(x)
        return h_n.squeeze(0)  # Return the last hidden state

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


class LSTMFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, n_features, hidden_dim):
        super().__init__(observation_space, features_dim=hidden_dim)
        self.lstm_encoder = LSTM(n_features, hidden_dim)

    def forward(self, observations):
        return self.lstm_encoder(observations)


class CustomLSTMPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        n_features=8,
        hidden_dim=64,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=LSTMFeaturesExtractor,
            features_extractor_kwargs=dict(
                n_features=n_features, hidden_dim=hidden_dim
            ),
            **kwargs,
        )
