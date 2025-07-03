import os
from stable_baselines3 import A2C, PPO
from components.environment import make_oc_atari_env
from components.wrappers import OCAtariEncoderWrapper
from components.policies.naive_agent import NaiveAgent
from components.agent_mappings import get_agent_mapping
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.env_util import make_atari_env
import gymnasium as gym
from ocatari.core import OCAtari
import yaml
import cv2
import argparse


def test(args):
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_path = "./weights"
    video_folder = "./videos/"
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    debug_video_path = os.path.join(video_folder, args.agent + ".mp4")
    upscale_factor = 4  # Change as needed for visibility
    frame_height = 210 if args.agent != "cnn" else 84  # Atari frame height
    frame_width = 160 if args.agent != "cnn" else 84  # Atari

    game_name = config["environment"]["game_name"]
    seed = args.seed
    model_name = config["model"]["name"]

    agent_mapping = get_agent_mapping(
        args.agent,
        game_name=game_name,
        model_name=model_name,
        model_extension=args.model,
    )
    wrapper_kwargs = {"clip_reward": False, "terminal_on_life_loss": False}

    if args.agent == "cnn":
        env = make_atari_env(
            game_name,
            n_envs=1,  # Single environment for evaluation
            seed=0,  # Fixed seed for reproducibility
            wrapper_kwargs=wrapper_kwargs,
        )
        env = VecTransposeImage(env)
        env = VecFrameStack(env, n_stack=4)
    else:
        oc_atari_kwargs = {
            "mode": "vision",
            "hud": False,
            "obs_mode": "ori",
        }
        env = make_oc_atari_env(
            game_name,
            n_envs=1,
            seed=0,
            env_kwargs=oc_atari_kwargs,
            wrapper_kwargs=wrapper_kwargs,
        )
    if agent_mapping["encoder"]:
        env = OCAtariEncoderWrapper(
            env,
            config["encoder"]["max_objects"],
            num_envs=1,
            speed_scale=config["encoder"]["speed_scale"],
        )

    if args.agent == "naive":
        model = NaiveAgent()
    else:
        # Load model
        if model_name == "A2C":
            model = A2C.load(
                os.path.join(model_path, agent_mapping["name"]),
                env=env,
                seed=seed,
                custom_objects={
                    "observation_space": env.observation_space,
                    "action_space": env.action_space,
                },
            )
        elif model_name == "PPO":
            model = PPO.load(
                os.path.join(model_path, agent_mapping["name"]),
                env=env,
                seed=seed,
                custom_objects={
                    "observation_space": env.observation_space,
                    "action_space": env.action_space,
                },
            )
        else:
            raise ValueError(f"Model {model_name} not implemented.")

    model.set_random_seed(seed)
    env.seed(seed)
    obs = env.reset()
    print(obs.shape)
    done = [False]

    # Prepare video writer for encoder visualization
    out = cv2.VideoWriter(
        debug_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,  # FPS
        (frame_width * upscale_factor, frame_height * upscale_factor),
        isColor=False if args.agent == "cnn" else True,
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
    image = obs if args.agent == "cnn" else info["image"]

    # Safety mechanisms to prevent infinite loops
    max_steps_per_life = 2000  # Maximum steps per life

    while not done[0]:
        # Visualize frame
        vis_frame = visualize_model(
            image,
            get_ocatari_objects(env.envs[0]),
            args.agent,
            upscale_factor=upscale_factor,
            frame_height=frame_height,
            frame_width=frame_width,
        )

        out.write(vis_frame)

        actions, _ = model.predict(obs, deterministic=args.deterministic)
        action = actions[0]
        obs, reward, done, info = env.step(actions)
        step_count += 1
        total_reward += reward[0]
        per_life_step += 1

        if isinstance(info, list):
            info = info[0]
        new_lives = info["lives"]
        image = obs if args.agent == "cnn" else info["image"]

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
    image,
    objects,
    agent,
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
    vis_frame = image.copy()
    if agent == "cnn":
        vis_frame = vis_frame[:, :, 0]  # Convert to grayscale if stacked
    else:
        for obj in objects:
            if obj.category != "NoObject":
                # Draw object bounding box
                x1 = int(obj.x)
                y1 = int(obj.y)
                x2 = int(obj.x + obj.w)
                y2 = int(obj.y + obj.h)
                cv2.rectangle(
                    vis_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),  # Green color for bounding box
                    1,
                )

    # Upscale for visibility
    vis_frame = cv2.resize(
        vis_frame,
        (frame_width * upscale_factor, frame_height * upscale_factor),
        interpolation=cv2.INTER_NEAREST,
    )

    # Draw object labels on upscaled frame for better text quality
    if agent != "cnn":
        for obj in objects:
            if obj.category != "NoObject":
                # Scale coordinates for upscaled frame
                x1_scaled = int(obj.x * upscale_factor)
                y1_scaled = int(obj.y * upscale_factor)

                # Draw object label on upscaled frame
                cv2.putText(
                    vis_frame,
                    f"{obj.category}",
                    (x1_scaled, y1_scaled - 1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.25 * upscale_factor,  # Scale font size with upscale factor
                    (255, 0, 255),  # Magenta color for label
                    max(1, upscale_factor // 2),  # Scale line thickness
                )

    return vis_frame


def get_ocatari_objects(env: gym.Env):
    """
    Get the objects from the underlying OCAtari environment through the wrapper chain.

    :param env: The wrapped environment
    :return: The objects from the OCAtari environment
    """
    # Traverse through wrappers to find the OCAtari instance
    current_env = env
    while hasattr(current_env, "env"):
        if isinstance(current_env, OCAtari):
            return current_env.objects
        current_env = current_env.env

    # Check if the current env is OCAtari
    if isinstance(current_env, OCAtari):
        return current_env.objects

    # If we can't find OCAtari, raise an error
    raise ValueError("No OCAtari environment found in the wrapper chain")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Rule-Based Encoder")
    parser.add_argument(
        "--agent",
        type=str,
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
