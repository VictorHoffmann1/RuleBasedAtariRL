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
        self.actor = nn.Linear(256, self.num_actions)
        self.critic = nn.Linear(256, 1)

    def forward(self, features):
        shared = self.shared(features)
        logits = self.actor(shared)
        value = self.critic(shared)
        return logits, value

    def act(self, logits, epsilon=0.0):
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

        # Generate random actions for exploration (entire batch)
        if epsilon > 0:
            random_mask = torch.rand(probs.shape[0], device=logits.device) < epsilon
            random_actions = torch.randint(
                0, self.num_actions, 
                (probs.shape[0], 1),  # shape: [num_envs, 1]
                device=logits.device
            )
            
            # Sample actions from policy
            policy_actions = dist.sample().unsqueeze(-1)  # shape: [num_envs, 1]
            
            # Combine random and policy actions
            actions = torch.where(
                random_mask.unsqueeze(-1),
                random_actions,
                policy_actions
            )
        else:
            actions = dist.sample().unsqueeze(-1)  # shape: [num_envs, 1]

        return actions, dist, probs


