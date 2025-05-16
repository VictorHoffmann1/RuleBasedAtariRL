def compute_returns(rewards, dones, last_value, gamma=0.99):
    returns = []
    R = last_value
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * (1 - dones[step])
        returns.insert(0, R)
    return returns