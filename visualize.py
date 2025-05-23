import os
from stable_baselines3 import A2C, PPO
from environment import make_atari_env
from wrappers import EncoderWrapper
from encoder import RuleBasedEncoder
import yaml
import cv2
import os

def test():

    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    game_name = config["environment"]["game_name"]
    seed = config["environment"]["seed"]
    n_features = config["model"]["n_features"]
    model_name = config["model"]["name"]

    model_path = "./weights"
    video_folder = "./videos/"
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    debug_video_path = os.path.join(video_folder, "encoder_debug.mp4")
    upscale_factor = 4  # Change as needed for visibility
    x_offset = 8
    y_offset = 31

    encoder = RuleBasedEncoder(**config["encoder"])

    wrapper_kwargs = {
        "greyscale": False,
        "screen_size": -1,
        "clip_reward": False,
        "terminal_on_life_loss": False,
        "max_pool": False,
    }

    env = make_atari_env(game_name, n_envs=1, seed=seed, wrapper_kwargs=wrapper_kwargs)
    encoder_env = EncoderWrapper(env, encoder, n_features)

    # Load model
    if model_name == "A2C":
        model = A2C.load(os.path.join(model_path, "a2c_breakout_rb"), env=encoder_env, seed=seed,
                        custom_objects={'observation_space': encoder_env.observation_space, 'action_space': encoder_env.action_space})
    elif model_name == "PPO":
        model = PPO.load(os.path.join(model_path, "ppo_breakout_rb"), env=encoder_env, seed=seed,
                    custom_objects={'observation_space': encoder_env.observation_space, 'action_space': encoder_env.action_space})
    else:
        raise ValueError(f"Model {model_name} not implemented.")

    obs = env.reset()
    done = [False]

    # Prepare video writer for encoder visualization
    frame_height, frame_width, _ = obs[0].shape
    out = cv2.VideoWriter(
        debug_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        30,  # FPS
        (frame_width * upscale_factor, frame_height * upscale_factor)
    )

    print("Starting test...")
    total_reward = 0
    step_count = 0
    while not done[0]:
        features = encoder(obs)
        actions, _ = model.predict(features)
        action = actions[0]

        # --- Visualization ---
        vis_frame = obs[0].copy()
        h, w = vis_frame.shape[:2]
        # Extract encoder features for visualization
        player_x = features[0][0]  # normalized [-1, 1]
        ball_x = features[0][1]    # normalized [-1, 1]
        ball_y = features[0][2]    # normalized [-1, 1]
        # Bricks: shape (num_brick_layers, num_bricks_per_layer)
        #bricks_flat = features[0][5:]
        #bricks = bricks_flat.reshape(encoder.num_brick_layers, encoder.num_bricks_per_layer)

        # Convert normalized positions to pixel coordinates
        px = int((player_x + 1) / 2 * (w - 2 * x_offset)) + x_offset
        bx = int((ball_x + 1) / 2 * (w - 2 * x_offset)) + x_offset
        by = int((ball_y + 1) / 2 * (h - y_offset)) + y_offset

        # Draw player position (blue circle)
        cv2.circle(vis_frame, (px, 189), 1, (255, 0, 0), -1)
        # Draw ball position (red circle)
        cv2.circle(vis_frame, (bx, by), 1, (0, 0, 255), -1)

        # Draw bricks (green rectangles)
        # brick_y0 = encoder.bricks_y_zone[0]
        # brick_h = encoder.brick_y_length
        # brick_w = encoder.brick_x_length
        # for i in range(encoder.num_brick_layers):
        #     for j in range(encoder.num_bricks_per_layer):
        #         if bricks[i, j]:
        #             x0 = j * brick_w + x_offset
        #             y0 = brick_y0 + i * brick_h + y_offset
        #             x1 = x0 + brick_w
        #             y1 = y0 + brick_h
        #             cv2.rectangle(vis_frame, (x0, y0), (x1, y1), (0, 255, 0), 1)

        # Upscale for visibility
        vis_frame = cv2.resize(vis_frame, (frame_width * upscale_factor, frame_height * upscale_factor), interpolation=cv2.INTER_NEAREST)
        out.write(vis_frame)
        # --- End visualization ---
        obs, reward, done, _ = env.step(actions)
        step_count += 1
        total_reward += reward[0]
        print(f"Step: {step_count}, Action: {action}, Reward: {reward[0]}, Done: {done[0]}")
    print(f"Test finished! Total reward: {total_reward}. Steps: {step_count}")
    env.close()
    out.release()
    print(f"Encoder debug video saved to {debug_video_path}")

if __name__ == "__main__":
    test()