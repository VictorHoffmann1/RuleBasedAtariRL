import os
from stable_baselines3 import A2C, PPO
from components.environment import make_atari_env
from components.wrappers import EncoderWrapper
from components.encoder import RuleBasedEncoder
from components.naive_agent import NaiveAgent
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
import yaml
import cv2
import argparse


def test(args):
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    game_name = config["environment"]["game_name"]
    seed = config["environment"]["seed"]
    model_name = config["model"]["name"]

    agent_mappings = {
        "player+ball": {
            "encoding_method": "paddle+ball",
            "n_features": 5,
            "name": model_name + "_rb_player_ball",
            "n_stack": None,
        },
        "player+ball+bricks": {
            "encoding_method": "bricks+paddle+ball",
            "n_features": 113,
            "name": model_name + "_rb_player_ball_bricks",
            "n_stack": None,
        },
        "transformer": {
            "encoding_method": "transformer",
            "n_features": -1,
            "name": model_name + "_rb_transformer",
            "n_stack": 2,  # Stack frames for temporal encoding
        },
        "deep_sets": {
            "encoding_method": "transformer",
            "n_features": 8,
            "name": model_name + "_rb_deep_sets",
            "n_stack": 2,  # Stack frames for temporal encoding
        },
        "cnn": {
            "encoding_method": "cnn",
            "n_features": -1,
            "name": model_name + "_cnn",
            "n_stack": 4,  # Stack frames for CNN
        },
        "rule_based": {
            "encoding_method": "rule_based",
            "n_features": -1,
            "name": None,
            "n_stack": None,  # No stacking for rule-based agent
        },
    }

    n_features = agent_mappings[args.agent]["n_features"]
    config["encoder"]["encoding_method"] = agent_mappings[args.agent]["encoding_method"]

    model_path = "./weights"
    video_folder = "./videos/"
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    debug_video_path = os.path.join(video_folder, args.agent + ".mp4")
    upscale_factor = 4  # Change as needed for visibility
    x_offset = 8
    y_offset = 31

    encoder = RuleBasedEncoder(**config["encoder"])

    if args.agent == "cnn":
        wrapper_kwargs = {}
    else:
        wrapper_kwargs = {
            "greyscale": True if args.agent in ["transformer", "deep_sets"] else False,
            "screen_size": -1,
            "clip_reward": False,
            "terminal_on_life_loss": False,
            "max_pool": False,
        }

    env = make_atari_env(game_name, n_envs=1, seed=seed, wrapper_kwargs=wrapper_kwargs)

    if args.agent == "cnn":
        env = VecFrameStack(env, n_stack=agent_mappings[args.agent]["n_stack"])
        env = VecTransposeImage(env)  # Transpose for CNN input
        load_env = env
    else:
        if args.agent in ["transformer", "deep_sets"]:
            env = VecFrameStack(env, n_stack=agent_mappings[args.agent]["n_stack"])
        load_env = EncoderWrapper(env, encoder, n_features)

    if args.agent == "rule_based":
        model = NaiveAgent()
    else:
        # Load model
        if model_name == "A2C":
            model = A2C.load(
                os.path.join(model_path, agent_mappings[args.agent]["name"]),
                env=load_env,
                seed=seed,
                custom_objects={
                    "observation_space": load_env.observation_space,
                    "action_space": load_env.action_space,
                },
            )
        elif model_name == "PPO":
            model = PPO.load(
                os.path.join(model_path, agent_mappings[args.agent]["name"]),
                env=load_env,
                seed=seed,
                custom_objects={
                    "observation_space": load_env.observation_space,
                    "action_space": load_env.action_space,
                },
            )
        else:
            raise ValueError(f"Model {model_name} not implemented.")

    obs = env.reset()
    done = [False]

    # Prepare video writer for encoder visualization
    frame_height, frame_width, _ = obs[0].shape
    out = cv2.VideoWriter(
        debug_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,  # FPS
        (frame_width * upscale_factor, frame_height * upscale_factor),
    )

    print("Starting test...")
    total_reward = 0
    step_count = 0
    while not done[0]:
        features = obs if args.agent == "cnn" else encoder(obs)
        actions, _ = model.predict(features, deterministic=True)
        action = actions[0]

        # --- Visualization ---
        vis_frame = obs[0].copy()
        if args.agent in ["player+ball", "player+ball+bricks", "rule_based"]:
            h, w = vis_frame.shape[:2]
            # Extract encoder features for visualization
            player_x = features[0][0]  # normalized [-1, 1]
            ball_x = features[0][1]  # normalized [-1, 1]
            ball_y = features[0][2]  # normalized [-1, 1]

            # Convert normalized positions to pixel coordinates
            px = int((player_x + 1) / 2 * (w - 2 * x_offset)) + x_offset
            bx = int((ball_x + 1) / 2 * (w - 2 * x_offset)) + x_offset
            by = int((ball_y + 1) / 2 * (h - y_offset)) + y_offset

            # Draw player position (blue circle)
            cv2.circle(vis_frame, (px, 189), 1, (255, 0, 0), -1)
            # Draw ball position (red circle)
            cv2.circle(vis_frame, (bx, by), 1, (0, 0, 255), -1)

            if args.agent == "player+ball+bricks":
                # Bricks: shape (num_brick_layers, num_bricks_per_layer)
                bricks_flat = features[0][5:]
                bricks = bricks_flat.reshape(
                    encoder.num_brick_layers, encoder.num_bricks_per_layer
                )

                # Draw bricks (green rectangles)
                brick_y0 = encoder.bricks_y_start
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
        vis_frame = cv2.resize(
            vis_frame,
            (frame_width * upscale_factor, frame_height * upscale_factor),
            interpolation=cv2.INTER_NEAREST,
        )
        out.write(vis_frame)
        # --- End visualization ---
        obs, reward, done, _ = env.step(actions)
        step_count += 1
        total_reward += reward[0]
        print(
            f"Step: {step_count}, Action: {action}, Reward: {reward[0]}, Done: {done[0]}"
        )
    print(f"Test finished! Total reward: {total_reward}. Steps: {step_count}")
    env.close()
    out.release()
    print(f"Encoder debug video saved to {debug_video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Rule-Based Encoder")
    parser.add_argument(
        "--agent",
        type=str,
        default="player+ball",
        choices=[
            "player+ball",
            "player+ball+bricks",
            "transformer",
            "cnn",
            "rule_based",
            "deep_sets",
        ],
        required=True,
        help="The agent type to test.",
    )
    args = parser.parse_args()
    test(args)
