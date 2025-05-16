import torch
import numpy as np
from components.model import ActorCriticMLP
from components.encoder import RuleBasedEncoder
from components.environment import create_environment
import yaml

def test():

    # Load configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the environment (use render_mode="human" to see it)
    env = create_environment(
        game_name=config["environment"]["game_name"],
        render_mode="rgb_array",
        record_video=True,
        video_dir="./data"
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

    model.load_state_dict(torch.load("./data/rulebased_a2c_breakout.pth", map_location=device, weights_only=True))
    model.eval()

    obs, _ = env.reset()
    # Just do one step for the ball to appear
    obs, _, _, _, _ = env.step(1)
    done = False

    print("Starting test...")
    step_count = 0
    while not done:
        with torch.no_grad():
            feature_space = torch.tensor(encoder(obs), dtype=torch.float32).unsqueeze(0).to(device)
            logits, _ = model(feature_space)
            probs = torch.softmax(logits, dim=1)
            action = torch.argmax(probs, dim=1).item()

        obs, reward, terminated, truncated, _ = env.step(action)
        step_count += 1
        print(f"Step: {step_count}, Action: {action}, Reward: {reward}")
        done = terminated or truncated
    print("Test finished.")
    env.close()

if __name__ == "__main__":
    test()