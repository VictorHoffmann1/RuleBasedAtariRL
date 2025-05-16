from components.environment import create_environment
from components.model import ActorCriticMLP
from components.helper import compute_returns
from components.encoder import RuleBasedEncoder
import pandas as pd
import torch
import tqdm
import yaml
import numpy as np

def train():

    # Load configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_envs = config["environment"]["number"]
    env = create_environment(game_name=config["environment"]["game_name"],
                                  render_mode="rgb_array",
                                  num_envs=num_envs)

    encoder = RuleBasedEncoder(
        num_brick_layers=config["encoder"]["num_brick_layers"],
        num_bricks_per_layer=config["encoder"]["num_bricks_per_layer"],
        bricks_y_zone=(config["encoder"]["bricks_start"],
                        config["encoder"]["bricks_end"]),
        frame_x_size=config["encoder"]["frame_x_size"],
        num_envs=num_envs
    )

    model = ActorCriticMLP(n_input=config["model"]["feature_space_size"],
                        num_actions=env.action_space[0].n).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(),
                                lr=config["optimizer"]["params"]["lr"],
                                weight_decay=config["optimizer"]["params"]["weight_decay"])

    max_updates = config["training"]["max_updates"]
    steps_per_update = config["training"]["steps_per_update"]
    gamma = config["training"]["gamma"]
    actor_weight = config["training"]["actor_weight"]
    critic_weight = config["training"]["critic_weight"]
    entropy_weight = config["training"]["entropy_weight"]
    update_frequency = config["training"]["update_frequency"]
    epsilon = config["training"]["epsilon_start"]
    epsilon_decay = config["training"]["epsilon_decay"]
    epsilon_end = config["training"]["epsilon_end"]
    seed = config["environment"]["seed"]

    states, _ = env.reset(seed=seed)
    training_data = []
    episode_returns = []
    current_episode_returns = np.zeros(num_envs)

    # --- Training Loop ---
    model.train()
    for update in tqdm.tqdm(range(max_updates)):
        log_probs_list = torch.zeros((steps_per_update, num_envs), device=device)
        values_list = torch.zeros((steps_per_update, num_envs, 1), device=device)
        rewards_list = torch.zeros((steps_per_update, num_envs), device=device)
        dones_list = torch.zeros((steps_per_update, num_envs), device=device)
        entropies_list = torch.zeros((steps_per_update), device=device)

        for step in range(steps_per_update):
            # Get policy and value from Actor-Critic model
            feature_space = torch.tensor(encoder(states), dtype=torch.float32).to(device)  # Shape: [num_envs, feature_dim]
            logits, values = model(feature_space)  # logits: [num_envs, num_actions], values: [num_envs, 1]
            actions, dist, probs = model.act(logits, epsilon)  # actions: [num_envs, 1]

            # Take action in the environment
            next_states, rewards, dones, _, _ = env.step(actions.squeeze(-1).cpu().numpy())

            # Store batched data
            log_probs_list[step] = dist.log_prob(actions.squeeze(-1))
            values_list[step] = values
            rewards_list[step] = torch.tensor(rewards, device=device)
            dones_list[step] = torch.tensor(dones, device=device)
            entropies_list[step] = dist.entropy().mean()

            # Track episode returns
            current_episode_returns += rewards

            # Reset terminated environments
            if dones.any():
                reset_indices = np.where(dones)[0]  # Indices of finished envs
                episode_returns = episode_returns + current_episode_returns[reset_indices].tolist()
                current_episode_returns[reset_indices] = 0
                encoder.reset(reset_indices)

            states = next_states

        # Batched advantage computation
        with torch.no_grad():
            last_features = torch.tensor(encoder(states), dtype=torch.float32).to(device)
            _, last_values = model(last_features)

        # Compute returns
        returns = compute_returns(rewards_list, dones_list, last_values, gamma)

        # Concatenate all stored data
        log_probs_list = log_probs_list.view(-1)  # [num_envs * steps_per_update]
        values_list = values_list.view(-1)        # [num_envs * steps_per_update]
        returns_list = returns.view(-1)                 # [num_envs * steps_per_update]

        # Compute advantages (now batched)
        advantages = returns_list - values_list
        advantages = (advantages - advantages.mean(dim=0)) / (advantages.std(dim=0) + 1e-8)
        
        # Decrease epsilon (exploration rate)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Compute loss
        actor_loss = -(log_probs_list * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        entropy_loss = entropies_list.mean()
        loss = actor_weight * actor_loss + critic_weight * critic_loss - entropy_weight * entropy_loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if update % update_frequency == 0:
            print("------ METRICS ------")
            avg_ep_return = sum(episode_returns) / len(episode_returns) if episode_returns else 0.0
            episode_returns = []
            torch.save(model.state_dict(), "./data/rulebased_a2c_breakout.pth")
            print(f"Update {update}: loss={loss.item():.3f}, avg_ep_return={avg_ep_return:.3f}")
            training_data.append({
                "update": update,
                "loss": loss.item(),
                "avg_ep_return": avg_ep_return
            })
            print("------- DEBUG PRINTS -------")
            # Debug print
            print("Action:", actions.view(-1).tolist(), "Avg. Prob:", np.round(probs.mean(dim=0).detach().cpu().numpy(), 3))
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"Grad norm: {total_norm:.4f}")
            print(f"Epsilon: {epsilon:.2f}")

    df = pd.DataFrame(training_data)
    df.to_csv("./data/training_log.csv", index=False)

    env.close()

if __name__ == "__main__":
    train()