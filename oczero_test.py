import argparse
import os

import cv2
import gymnasium as gym
import numpy as np
import torch
import yaml
from ocatari.core import OCAtari

from components.agent_mappings import get_agent_mapping
from components.agents.OCZero.env_model import EnvModel
from components.utils import create_env, load_model


def oc_zero_test(args):
    num_envs = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_path = "./weights"
    video_folder = "./videos/"
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    debug_video_path = os.path.join(video_folder, "OC_ZERO_TEST" + ".mp4")
    upscale_factor = 4  # Change as needed for visibility
    frame_height = 210
    frame_width = 160

    game_name = config["environment"]["game_name"]
    seed = args.seed

    agent_mapping = get_agent_mapping(
        "relational_network", game_name=game_name, model_extension="10M"
    )

    env = create_env(
        config,
        agent_mapping,
        n_envs=num_envs,
        seed=seed,
        train=True,
    )

    model = load_model(
        env, agent_mapping, os.path.join(model_path, agent_mapping["name"])
    )

    model_env = EnvModel(n_features=6, n_actions=env.action_space.n).to(device)
    optimizer = torch.optim.Adam(model_env.parameters(), lr=1e-4)

    model.set_random_seed(seed)
    env.seed(seed)
    obs = env.reset()

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
    # Get initial lives from info dict after first step
    obs, reward, done, infos = env.step([1] * num_envs)  # Force Fire action
    info = infos[0] if isinstance(infos, list) else infos
    image = info["image"]

    while True:
        # # Visualize frame
        # vis_frame = visualize_model(
        #     image,
        #     get_ocatari_objects(env.envs[0]),
        #     args.agent,
        #     upscale_factor=upscale_factor,
        #     frame_height=frame_height,
        #     frame_width=frame_width,
        # )
        # out.write(vis_frame)

        actions, _ = model.predict(obs, deterministic=args.deterministic)

        # environment model training
        th_obs = torch.tensor(obs, dtype=torch.float32, device=device)
        th_actions = torch.tensor(
            actions, dtype=torch.float32, device=device
        ).unsqueeze(1)

        predicted_next_obs = model_env(th_obs, th_actions)
        loss = model_env.compute_loss(
            predicted_next_obs, th_obs, iou_weight=1.0, l1_weight=1.0, bce_weight=1.0
        )
        print(
            f"Step {step_count}, Loss: {loss['total_loss'].item()}, L1 Pos Loss: {loss['l1_pos_loss'].item()}, L1 Speed Loss: {loss['l1_speed_loss'].item()}, L1 Shape Loss: {loss['l1_shape_loss'].item()}, BCE Loss: {loss['bce_loss'].item()}"
        )

        optimizer.zero_grad()
        loss["total_loss"].backward()
        optimizer.step()

        obs, reward, done, info = env.step(actions)
        info = info[0] if isinstance(info, list) else info
        step_count += 1
        total_reward += np.mean(reward)

        image = info["image"]

        if step_count % args.verbose_update == 0:
            print(f"Step: {step_count}, Current Average Score: {total_reward}")

    print(f"Test finished! Final Score: {total_reward}, Steps: {step_count}")
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

    parser.add_argument(
        "--verbose_update",
        type=int,
        default=100,
        help="Frequency of verbose updates during testing.",
    )
    args = parser.parse_args()
    oc_zero_test(args)
