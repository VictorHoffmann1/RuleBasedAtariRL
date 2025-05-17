import torch
import numpy as np
from components.model import ActorCriticMLP
from components.encoder import RuleBasedEncoder
from components.environment import create_environment
import yaml
import cv2

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

    encoder = RuleBasedEncoder(
        num_brick_layers=config["encoder"]["num_brick_layers"],
        num_bricks_per_layer=config["encoder"]["num_bricks_per_layer"],
        bricks_y_zone=(config["encoder"]["bricks_start"],
                        config["encoder"]["bricks_end"]),
        frame_x_size=config["encoder"]["frame_x_size"],
        num_envs=1
    )

    model = ActorCriticMLP(n_input=config["model"]["feature_space_size"],
                        num_actions=env.action_space.n).to(device)

    model.load_state_dict(torch.load("./data/rulebased_a2c_breakout_best.pth", map_location=device, weights_only=True))
    model.eval()

    obs, _ = env.reset()
    done = False

    # Prepare video writer for encoder visualization
    debug_video_path = "./data/encoder_debug.mp4"
    upscale_factor = 4  # Change as needed for visibility
    x_offset = 8
    y_offset = 31
    frame_height, frame_width, _ = obs.shape
    out = cv2.VideoWriter(
        debug_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        30,  # FPS
        (frame_width * upscale_factor, frame_height * upscale_factor)
    )

    print("Starting test...")
    total_reward = 0
    step_count = 0
    while not done:
        with torch.no_grad():
            features = encoder([obs])
            feature_space = torch.tensor(features, dtype=torch.float32).to(device)
            logits, _ = model(feature_space)
            actions, _, _ = model.act(logits)
            action = actions.item()

        # --- Visualization ---
        vis_frame = obs.copy()
        h, w = vis_frame.shape[:2]
        # Extract encoder features for visualization
        player_x = features[0][0]  # normalized [-1, 1]
        ball_x = features[0][1]    # normalized [-1, 1]
        ball_y = features[0][2]    # normalized [-1, 1]
        # Bricks: shape (num_brick_layers, num_bricks_per_layer)
        bricks_flat = features[0][5:]
        bricks = bricks_flat.reshape(encoder.num_brick_layers, encoder.num_bricks_per_layer)

        # Convert normalized positions to pixel coordinates
        px = int((player_x + 1) / 2 * w) + x_offset
        bx = int((ball_x + 1) / 2 * w) + x_offset
        by = int((ball_y + 1) / 2 * h) + y_offset

        # Draw player position (blue circle)
        cv2.circle(vis_frame, (px, h - 10), 1, (255, 0, 0), -1)
        # Draw ball position (red circle)
        cv2.circle(vis_frame, (bx, by), 1, (0, 0, 255), -1)

        # Draw bricks (green rectangles)
        brick_y0 = encoder.bricks_y_zone[0]
        brick_h = encoder.brick_y_length
        brick_w = encoder.brick_x_length
        for i in range(encoder.num_brick_layers):
            for j in range(encoder.num_bricks_per_layer):
                if bricks[i, j]:
                    x0 = j * brick_w + x_offset
                    y0 = brick_y0 + i * brick_h + y_offset
                    x1 = x0 + brick_w
                    y1 = y0 + brick_h
                    cv2.rectangle(vis_frame, (x0, y0), (x1, y1), (0, 255, 0), 1)

        # Upscale for visibility
        vis_frame = cv2.resize(vis_frame, (frame_width * upscale_factor, frame_height * upscale_factor), interpolation=cv2.INTER_NEAREST)
        out.write(vis_frame)
        # --- End visualization ---
        obs, reward, terminated, truncated, _ = env.step(action)
        step_count += 1
        total_reward += reward
        done = terminated or truncated
        print(f"Step: {step_count}, Action: {action}, Reward: {reward}, Done: {done}")
    print(f"Test finished! Total reward: {total_reward}. Steps: {step_count}")
    env.close()
    out.release()
    print(f"Encoder debug video saved to {debug_video_path}")

if __name__ == "__main__":
    test()