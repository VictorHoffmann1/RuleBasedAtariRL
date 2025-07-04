import os
from stable_baselines3 import A2C, PPO
from components.environment import make_atari_env
from components.wrappers import EncoderWrapper
from components.naive_agent import NaiveAgent
from components.agent_mappings import get_agent_mapping
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
import yaml
import cv2
import argparse


def test(args):
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    game_name = config["environment"]["game_name"]
    seed = args.seed
    model_name = config["model"]["name"]

    agent_mapping = get_agent_mapping(
        args.agent,
        config,
        n_envs=1,  # Single environment for testing
        game_name=game_name,
        model_name=model_name,
        model_extension=args.model,
    )

    model_path = "./weights"
    video_folder = "./videos/"
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    debug_video_path = os.path.join(video_folder, args.agent + ".mp4")
    upscale_factor = 4  # Change as needed for visibility
    x_offset = 8
    y_offset = 31

    agent_mapping["wrapper_kwargs"]["clip_reward"] = False
    agent_mapping["wrapper_kwargs"]["terminal_on_life_loss"] = False

    encoder = agent_mapping["encoder"]
    env = make_atari_env(
        game_name, n_envs=1, seed=seed, wrapper_kwargs=agent_mapping["wrapper_kwargs"]
    )

    if args.agent == "cnn":
        env = VecTransposeImage(env)  # Transpose for CNN input
    if agent_mapping["n_stack"] is not None:
        env = VecFrameStack(env, n_stack=agent_mapping["n_stack"])
    if encoder is not None:
        load_env = EncoderWrapper(
            env,
            encoder,
            agent_mapping["n_features"],
        )
    else:
        load_env = env

    if args.agent == "naive":
        model = NaiveAgent()
    else:
        # Load model
        if model_name == "A2C":
            model = A2C.load(
                os.path.join(model_path, agent_mapping["name"]),
                env=load_env,
                seed=seed,
                custom_objects={
                    "observation_space": load_env.observation_space,
                    "action_space": load_env.action_space,
                },
            )
        elif model_name == "PPO":
            model = PPO.load(
                os.path.join(model_path, agent_mapping["name"]),
                env=load_env,
                seed=seed,
                custom_objects={
                    "observation_space": load_env.observation_space,
                    "action_space": load_env.action_space,
                },
            )
        else:
            raise ValueError(f"Model {model_name} not implemented.")

    # Make sure the seed is properly set
    env.seed(seed)
    load_env.seed(seed)
    model.set_random_seed(seed)

    obs = env.reset()
    done = [False]

    # Prepare video writer for encoder visualization
    frame_height, frame_width, channel = obs[0].shape
    out = cv2.VideoWriter(
        debug_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,  # FPS
        (frame_width * upscale_factor, frame_height * upscale_factor),
        isColor=True if channel == 3 else False,
    )

    print("Starting test...")
    total_reward = 0
    step_count = 0
    per_life_step = 0
    # Get initial lives from info dict after first step
    obs, reward, done, info = env.step([1])  # Force Fire action
    if isinstance(info, list):
        info = info[0]
    lives = info.get("lives", None)

    # Safety mechanisms to prevent infinite loops
    max_steps = 10000  # Maximum steps per episode
    max_steps_per_life = 2000  # Maximum steps per life

    while not done[0]:
        # Visualize frame
        features = obs if encoder is None else encoder(obs)

        vis_frame = visualize_model(
            obs,
            features,
            encoder,
            args.agent,
            agent_mapping,
            x_offset=x_offset,
            y_offset=y_offset,
            upscale_factor=upscale_factor,
            frame_height=frame_height,
            frame_width=frame_width,
        )

        out.write(vis_frame)

        actions, _ = model.predict(features, deterministic=args.deterministic)
        action = actions[0]
        obs, reward, done, info = env.step(actions)
        step_count += 1
        total_reward += reward[0]
        per_life_step += 1

        new_lives = info[0].get("lives", lives)
        if new_lives < lives:
            obs, _, _, info = env.step([1])  # Force Fire action
            per_life_step = 0
        elif per_life_step > max_steps_per_life:
            # Safety: Force end if stuck on same life too long
            print(
                f"  Warning: Forced termination after {per_life_step} steps on life {lives}"
            )
            break
        print(
            f"Step: {step_count}, Action: {action}, Reward: {reward[0]}, Done: {done[0]}"
        )
        lives = new_lives
    print(f"Test finished! Total reward: {total_reward}. Steps: {step_count}")
    env.close()
    out.release()
    print(f"Encoder debug video saved to {debug_video_path}")


def visualize_model(
    obs,
    features,
    encoder,
    agent,
    agent_mapping,
    x_offset=8,
    y_offset=31,
    upscale_factor=4,
    frame_height=210,
    frame_width=160,
):
    """
    Visualize the model's encoder features on the observation frame.
    Args:
        obs: Current observation from the environment.
        features: Encoder features to visualize.
        encoder: The encoder used to extract features.
    """
    # --- Visualization ---
    vis_frame = obs[0].copy()
    if agent in ["player+ball", "player+ball+bricks", "naive"]:
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

        if "bricks" in agent:
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

    if agent_mapping["n_stack"] is not None:
        vis_frame = vis_frame[:, :, 0]  # Convert to grayscale if stacked

    # Upscale for visibility
    vis_frame = cv2.resize(
        vis_frame,
        (frame_width * upscale_factor, frame_height * upscale_factor),
        interpolation=cv2.INTER_NEAREST,
    )

    return vis_frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Rule-Based Encoder")
    parser.add_argument(
        "--agent",
        type=str,
        default="player+ball",
        required=True,
        help="The agent type to test.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="The model type to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--deterministic",
        type=bool,
        default=True,
        help="Use deterministic actions for evaluation.",
    )
    args = parser.parse_args()
    test(args)
