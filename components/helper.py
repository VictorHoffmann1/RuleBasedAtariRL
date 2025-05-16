import torch

def compute_returns(rewards, dones, last_values, gamma):
    # rewards: (steps, num_envs)
    # dones: (steps, num_envs)
    # last_values: (num_envs,)
    steps, num_envs = rewards.shape
    returns = torch.zeros_like(rewards)
    R = last_values.squeeze(-1)  # (num_envs)
    for t in reversed(range(steps)):
        R = rewards[t] + gamma * R * (1 - dones[t])
        returns[t] = R
    return returns  # (steps, num_envs)