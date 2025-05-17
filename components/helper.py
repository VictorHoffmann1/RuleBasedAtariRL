import torch

def compute_returns(rewards, dones, last_values, gamma):
    # rewards: (steps, num_envs)
    # dones: (steps, num_envs)
    # last_values: (num_envs,)
    steps, num_envs = rewards.shape
    returns = torch.zeros_like(rewards)
    R = last_values  # (num_envs)
    for t in reversed(range(steps)):
        R = rewards[t] + gamma * R * (1 - dones[t])
        returns[t] = R
    return returns  # (steps, num_envs)

def compute_gae(rewards, dones, values, last_values, gamma, lam):
    """
    rewards: (steps, num_envs)
    dones: (steps, num_envs)
    values: (steps, num_envs)
    last_values: (num_envs,)
    gamma: discount factor
    lam: GAE lambda (typically 0.95)
    """
    steps, num_envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(num_envs, device=rewards.device)
    
    values = torch.cat([values, last_values.unsqueeze(0)], dim=0)  # (steps + 1, num_envs)

    for t in reversed(range(steps)):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values[:-1]  # (steps, num_envs)
    return returns, advantages