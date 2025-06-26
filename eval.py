from stable_baselines3 import A2C, PPO
from components.environment import make_atari_env
from components.wrappers import EncoderWrapper
from components.naive_agent import NaiveAgent
from components.agent_mappings import get_agent_mapping
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

import yaml
import os
import numpy as np
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def eval(
    model=None,
    agent: str = "unknown",
    model_extension: str = "eval",
    deterministic: bool = True,
    verbose: bool = False,
):
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    game_name = config["environment"]["game_name"]
    model_name = config["model"]["name"]
    model_path = "./weights"

    agent_mapping = get_agent_mapping(
        agent,
        config,
        n_envs=1,  # Single environment for evaluation
        game_name=game_name,
        model_name=model_name,
        model_extension=model_extension,
    )

    agent_mapping["wrapper_kwargs"]["clip_reward"] = False
    agent_mapping["wrapper_kwargs"]["terminal_on_life_loss"] = False

    env = make_atari_env(
        game_name,
        n_envs=1,
        seed=0,
        wrapper_kwargs=agent_mapping["wrapper_kwargs"],
    )
    if agent == "cnn":
        env = VecTransposeImage(env)
    if agent_mapping["n_stack"] is not None:
        # Stack frames to encode temporal information
        env = VecFrameStack(env, n_stack=agent_mapping["n_stack"])
    if agent_mapping["encoder"] is not None:
        env = EncoderWrapper(
            env,
            agent_mapping["encoder"],
            agent_mapping["n_features"],
        )

    if model is None:
        if agent == "naive":
            model = NaiveAgent()
        else:
            # Load model
            if model_name == "A2C":
                model = A2C.load(
                    os.path.join(model_path, agent_mapping["name"]),
                    env=env,
                    seed=0,
                    custom_objects={
                        "observation_space": env.observation_space,
                        "action_space": env.action_space,
                    },
                )
            elif model_name == "PPO":
                model = PPO.load(
                    os.path.join(model_path, agent_mapping["name"]),
                    env=env,
                    seed=0,
                    custom_objects={
                        "observation_space": env.observation_space,
                        "action_space": env.action_space,
                    },
                )
            else:
                raise ValueError(f"Model {model_name} not implemented.")

    seeds = list(range(args.n_seeds))  # Use a range of seeds for evaluation
    total_rewards = []
    for seed in seeds:
        env.seed(seed)
        model.set_random_seed(seed)

        obs = env.reset()
        done = [False]

        total_reward = 0
        step_count = 0
        # --- Track per-life stats ---
        per_life_reward = 0
        per_life_step = 0
        # Get initial lives from info dict after first step
        obs, reward, done, info = env.step([1])  # Force Fire action
        if isinstance(info, list):
            info = info[0]
        lives = info.get("lives", None)

        # Safety mechanisms to prevent infinite loops
        max_steps = 10000  # Maximum steps per episode
        max_steps_per_life = 2000  # Maximum steps per life
        steps_since_life_change = 0

        while not done[0] and step_count < max_steps:
            actions, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(actions)
            step_count += 1
            steps_since_life_change += 1
            total_reward += reward[0]
            per_life_reward += reward[0]
            per_life_step += 1

            if isinstance(info, list):
                info = info[0]
            new_lives = info.get("lives", lives)

            # Check for life change
            if new_lives < lives:
                if verbose and steps_since_life_change > 2000:
                    print(
                        f"  Life lost after {steps_since_life_change} steps (life {lives} -> {new_lives})"
                    )
                obs, _, _, info = env.step([1])  # Force Fire action
                per_life_reward = 0
                per_life_step = 0
                steps_since_life_change = 0
            elif steps_since_life_change > max_steps_per_life:
                # Safety: Force end if stuck on same life too long
                if verbose:
                    print(
                        f"  Warning: Forced termination after {steps_since_life_change} steps on life {lives}"
                    )
                break
            lives = new_lives
        if verbose:
            print(f"Seed {seed}: Reward: {total_reward}. Steps: {step_count}")
        env.close()
        total_rewards.append(total_reward)

    total_rewards = np.array(total_rewards)
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)

    if verbose:
        print(f"Average Reward over {len(seeds)} seeds: {avg_reward}")
        print(f"Standard Deviation: {std_reward}")

    return avg_reward, std_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="The agent type to evaluate.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="The model type to evaluate.",
    )
    parser.add_argument(
        "--deterministic",
        type=bool,
        default=True,
        help="Run the evaluation in deterministic mode.",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=10,
        help="Number of random seeds to use for evaluation.",
    )
    args = parser.parse_args()
    eval(
        None,
        agent=args.agent,
        model_extension=args.model,
        deterministic=args.deterministic,
        verbose=True,
    )
