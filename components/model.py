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
            nn.Linear(n_input, 512),
            nn.ReLU()
        )
        self.actor = nn.Linear(512, self.num_actions)
        self.critic = nn.Linear(512, 1)

    def forward(self, features):
        shared = self.shared(features)
        logits = self.actor(shared)
        value = self.critic(shared)
        return logits, value

    def act(self, logits, epsilon=0.0):
        probs = torch.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)

        if random.random() < epsilon:
            # Take a random action
            action = torch.tensor([[random.randint(0, self.num_actions - 1)]], device=logits.device)
            return action.squeeze(), dist, probs  # Return dist anyway for consistency/logging
        else:
            action = dist.sample()
            return action.squeeze(), dist, probs


