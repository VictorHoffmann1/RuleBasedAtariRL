import argparse
import os

import cv2
import gymnasium as gym
import numpy as np
import torch
import yaml
from ocatari.core import OCAtari
from tqdm import tqdm

from components.agent_mappings import get_agent_mapping
from components.agents.OCZero.world_model import WorldModel
from components.utils import create_env, load_model


def oc_zero_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_path = "./weights"
    video_folder = "./videos/world_model"
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
        n_envs=args.num_envs,
        seed=seed,
        train=True,
    )

    model = load_model(
        env, agent_mapping, os.path.join(model_path, agent_mapping["name"])
    )

    world_model = WorldModel(n_features=6, n_actions=env.action_space.n).to(device)
    optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-3)

    model.set_random_seed(seed)
    env.seed(seed)
    obs = env.reset()

    # Prepare video writer for encoder visualization
    # out = cv2.VideoWriter(
    #     debug_video_path,
    #     cv2.VideoWriter_fourcc(*"mp4v"),
    #     30,  # FPS
    #     (frame_width * upscale_factor, frame_height * upscale_factor),
    #     isColor=True,
    # )

    print("Starting test...")
    total_reward = 0
    step_count = 0
    max_steps = 10000  # Calculate total steps for progress bar

    # Get initial lives from info dict after first step
    obs, reward, done, infos = env.step([0] * args.num_envs)
    th_obs = torch.tensor(obs, dtype=torch.float32, device=device)
    info = infos[0] if isinstance(infos, list) else infos
    image = info["image"]

    # Initialize progress bar
    pbar = tqdm(total=max_steps, desc="", unit="step")

    while step_count < max_steps:
        actions, _ = model.predict(obs, deterministic=args.deterministic)

        # environment model training
        th_actions = torch.tensor(
            actions, dtype=torch.float32, device=device
        ).unsqueeze(1)

        predicted_next_obs = world_model(th_obs, th_actions)

        next_obs, reward, done, info = env.step(actions)
        info = info[0] if isinstance(info, list) else info

        th_next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
        loss = world_model.compute_loss(
            predicted_next_obs,
            th_next_obs,
            use_iou=False,
            iou_weight=1.0,
            l1_weight=1.0,
            bce_weight=1.0,
        )

        optimizer.zero_grad()
        loss["total_loss"].backward()
        optimizer.step()

        total_reward += np.mean(reward)
        step_count += args.num_envs
        obs = next_obs.copy()
        th_obs = th_next_obs.clone()

        # Update progress bar with loss information
        pbar.set_postfix(
            {
                "Total": f"{loss['total_loss'].item():<.3f}",
                "IOU": f"{loss['iou_loss'].item():<.3f}",
                "Pos": f"{loss['l1_pos_loss'].item():<.3f}",
                "Speed": f"{loss['l1_speed_loss'].item():<.3f}",
                "Shape": f"{loss['l1_shape_loss'].item():<.3f}",
                "BCE": f"{loss['bce_loss'].item():<.3f}",
                "Rwrd": f"{total_reward:.0f}",
            }
        )
        pbar.update(args.num_envs)

    pbar.close()
    print(f"Training on {step_count:.0f} steps finished! Generating images...")

    env.seed(seed + 1)
    obs = env.reset()
    num_steps = 50  # Calculate total steps for progress bar

    # Get initial lives from info dict after first step
    obs, reward, done, infos = env.step([0] * args.num_envs)
    th_obs = torch.tensor(obs, dtype=torch.float32, device=device)
    info = infos[0] if isinstance(infos, list) else infos
    image = info["image"]

    for i in range(num_steps):
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

        th_actions = torch.tensor(
            actions, dtype=torch.float32, device=device
        ).unsqueeze(1)

        next_obs, reward, done, info = env.step(actions)
        info = info[0] if isinstance(info, list) else info

        image = info["image"]
        vis_frame = visualize_objects(
            image,
            get_ocatari_objects(env.envs[0]),
            world_model.get_objects(th_obs, th_actions)[0][0],
            upscale_factor=upscale_factor,
            frame_height=frame_height,
            frame_width=frame_width,
        )

        th_next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

        # Save the visualization frame
        cv2.imwrite(os.path.join(video_folder, f"frame_{i:04d}.png"), vis_frame)

        obs = next_obs.copy()
        th_obs = th_next_obs.clone()

    env.close()
    # out.release()
    print(f"Images Saved In Folder {video_folder}")


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


def visualize_objects(
    image,
    true_objects,
    pred_objects,
    upscale_factor=4,
    frame_height=210,
    frame_width=160,
):
    """
    Visualize the objects on the observation frame.
    Args:
        image: Current observation from the environment.
        objects: List of objects to visualize.
    """
    vis_frame = image.copy()
    for obj in true_objects:
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

    for obj in pred_objects:
        cx, cy, vx, vy, w, h = obj
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        cv2.rectangle(
            vis_frame,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),  # Red color for bounding box
            1,
        )

    # Upscale for visibility
    vis_frame = cv2.resize(
        vis_frame,
        (frame_width * upscale_factor, frame_height * upscale_factor),
        interpolation=cv2.INTER_NEAREST,
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
    parser.add_argument(
        "--num_envs",
        type=int,
        default=8,
        help="Number of environments to run in parallel.",
    )
    args = parser.parse_args()
    oc_zero_test(args)
