from components.environment import create_environment
from components.model import ActorCriticMLP
from components.helper import compute_returns
from components.encoder import RuleBasedEncoder
import pandas as pd
import torch
import tqdm
import yaml

def train():

    # Load configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = create_environment(
        game_name=config["environment"]["game_name"],
        noop_max=config["environment"]["noop_max"],
        render_mode="rgb_array",
        )
    
    encoder = RuleBasedEncoder(
        num_brick_layers=config["encoder"]["num_brick_layers"],
        num_bricks_per_layer=config["encoder"]["num_bricks_per_layer"],
        bricks_y_zone=(config["encoder"]["bricks_start"],
                        config["encoder"]["bricks_end"]),
        frame_x_size=config["encoder"]["frame_x_size"],
    )

    model = ActorCriticMLP(n_input=config["model"]["feature_space_size"],
                        num_actions=env.action_space.n).to(device)
    
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

    state, _ = env.reset(seed=seed)

    training_data = []
    episode_returns = []
    current_episode_return = 0.0

    # --- Training Loop ---
    model.train()
    for update in tqdm.tqdm(range(max_updates)):
        log_probs = []
        values = []
        rewards = []
        dones = []
        entropies = []

        for step in range(steps_per_update):
            # Get policy and value from Actor-Critic model
            feature_space = torch.tensor(encoder(state), dtype=torch.float32).unsqueeze(0).to(device)
            logits, value = model(feature_space)
            action, dist, probs = model.act(logits, epsilon)

            # Take action in the environment
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            log_probs.append(dist.log_prob(action).squeeze())
            values.append(value.view(-1))
            reward_tensor = torch.tensor(reward, dtype=torch.float32, device=device)
            rewards.append(reward_tensor)
            current_episode_return += reward
            dones.append(torch.tensor(done, dtype=torch.float32, device=device))
            entropies.append(dist.entropy().mean())

            if done:
                encoder.reset()
                episode_returns.append(current_episode_return)
                current_episode_return = 0.0

                next_state, _ = env.reset(seed=seed)

            state = next_state

        # Get the value of the next state
        with torch.no_grad():
            feature_space = torch.tensor(encoder(state), dtype=torch.float32).unsqueeze(0).to(device)
            _, last_value = model(feature_space)

        # Compute returns
        returns = compute_returns(rewards, dones, last_value.view(-1), gamma)
        returns = torch.stack(returns).detach()
        values = torch.cat(values)
        log_probs = torch.stack(log_probs)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)


        # Debug print
        print("Action:", action.item(), "Prob:", probs.squeeze().tolist())
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        print(f"Grad norm: {total_norm:.4f}")

        # Compute loss
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        entropy_loss = torch.stack(entropies).mean()
        loss = actor_weight * actor_loss + critic_weight * critic_loss - entropy_weight * entropy_loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        if update % update_frequency == 0:
            avg_ep_return = sum(episode_returns[-10:]) / len(episode_returns[-10:]) if episode_returns else 0.0
            torch.save(model.state_dict(), "./data/rulebased_a2c_breakout.pth")
            print(f"Update {update}: loss={loss.item():.3f}, avg_ep_return={avg_ep_return:.3f}")
            training_data.append({
                "update": update,
                "loss": loss.item(),
                "avg_ep_return": avg_ep_return
            })

    df = pd.DataFrame(training_data)
    df.to_csv("./data/training_log.csv", index=False)

    env.close()

if __name__ == "__main__":
    train()