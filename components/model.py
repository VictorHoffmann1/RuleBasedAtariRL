import torch
import torch.nn as nn
import random

class ActorCriticMLP(nn.Module):
    def __init__(self, 
                 n_input,
                 num_actions):
        super().__init__()

        # The input is already encoded with a rule-based encoder
        self.num_actions = num_actions
        self.shared = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.ReLU()
        )

        # Actor and Critic heads
        self.actor = nn.Linear(256, self.num_actions)
        self.critic = nn.Linear(256, 1)

        self._init_weights()

    def _init_weights(self):
        # He (Kaiming) initialization for all linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features):
        shared = self.shared(features)
        logits = self.actor(shared)
        value = self.critic(shared)
        return logits, value

    def act(self, logits):
        """
        Args:
            logits: Tensor of shape [num_envs, num_actions]
            epsilon: Exploration rate
        Returns:
            actions: Tensor of shape [num_envs, 1]
            dist: Categorical distribution
            probs: Tensor of shape [num_envs, num_actions]
        """
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)  # shape: [num_envs, num_actions]
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample().unsqueeze(-1)  # shape: [num_envs, 1]

        return actions, dist, probs


