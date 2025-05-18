import torch
import numpy as np
from components.dl_model import ActorCriticCNN
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
        video_dir="./data",
        num_envs=1
        )

    model = ActorCriticCNN(num_actions=env.action_space.n).to(device)

    model.load_state_dict(torch.load("./data/rulebased_a2c_breakout_best.pth", map_location=device, weights_only=True))
    model.eval()

    obs, _ = env.reset()
    done = False

    print("Starting test...")
    total_reward = 0
    step_count = 0
    while not done:
        with torch.no_grad():
            logits, _ = model([obs])
            actions, _, _ = model.act(logits)
            action = actions.item()

        obs, reward, terminated, truncated, _ = env.step(action)
        step_count += 1
        total_reward += reward
        done = terminated or truncated
        print(f"Step: {step_count}, Action: {action}, Reward: {reward}, Done: {done}")
    print(f"Test finished! Total reward: {total_reward}. Steps: {step_count}")
    env.close()
if __name__ == "__main__":
    test()