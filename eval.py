from stable_baselines3 import A2C, PPO
from components.environment import make_oc_atari_env
from components.wrappers import OCAtariEncoderWrapper
from components.policies.naive_agent import NaiveAgent
from components.agent_mappings import get_agent_mapping
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.env_util import make_atari_env
import yaml
import os
import numpy as np
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def eval(
    model=None,
    env=None,
    agent: str = "unknown",
    model_extension: str = "eval",
    n_seeds: int = 10,
    deterministic: bool = True,
    verbose: bool = False,
    return_lists: bool = False,
):
    n_envs = 1 if env is None else env.num_envs
    if env is None:
        # Load configuration
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        game_name = config["environment"]["game_name"]
        model_path = "./weights"

        agent_mapping = get_agent_mapping(agent, game_name, model_extension)
        wrapper_kwargs = {"clip_reward": False, "terminal_on_life_loss": False}

        if agent == "cnn":
            env = make_atari_env(
                game_name,
                n_envs=n_envs,  # Single environment for evaluation
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
                "frameskip": 4,
                "repeat_action_probability": 0.0,
            }
            env = make_oc_atari_env(
                game_name,
                n_envs=n_envs,
                seed=0,
                env_kwargs=oc_atari_kwargs,
                wrapper_kwargs=wrapper_kwargs,
            )
        if agent_mapping["encoder"]:
            env = OCAtariEncoderWrapper(
                env,
                config["encoder"]["max_objects"],
                num_envs=n_envs,
                method=agent_mapping["method"],
                speed_scale=config["encoder"]["speed_scale"],
                use_rgb=config["encoder"]["use_rgb"],
                use_category=config["encoder"]["use_category"],
            )

    if model is None:
        if agent == "naive":
            model = NaiveAgent()
        else:
            model = PPO.load(
                os.path.join(model_path, agent_mapping["name"]),
                env=env,
                seed=0,
                custom_objects={
                    "observation_space": env.observation_space,
                    "action_space": env.action_space,
                },
            )

    seeds = list(range(n_seeds))  # Use a range of seeds for evaluation
    total_rewards = []
    episode_lengths = []
    for seed in seeds:
        model.set_random_seed(seed)
        env.seed(seed)
        env.reset()

        done = [False]

        total_reward = 0
        step_count = 0
        per_life_step = 0
        # Get initial lives from info dict after first step
        obs, reward, done, info = env.step([1] * n_envs)  # Force Fire action
        if isinstance(info, list):
            info = info[0]
        lives = info.get("lives", None)

        # Safety mechanisms to prevent infinite loops
        max_steps = 10000  # Maximum steps per episode
        max_steps_per_life = 2000  # Maximum steps per life

        while not done[0] and step_count < max_steps:
            actions, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(actions)
            step_count += 1
            total_reward += reward[0]
            per_life_step += 1

            new_lives = info[0].get("lives", lives)

            # Check for life change
            if new_lives < lives:
                if agent != "cnn":  # Raises FrameStack error for CNN agent
                    obs, _, _, info = env.step([1] * n_envs)  # Force Fire action
                per_life_step = 0
            elif per_life_step > max_steps_per_life:
                # Safety: Force end if stuck on same life too long
                if verbose:
                    print(
                        f"  Warning: Forced termination after {per_life_step} steps on life {lives}"
                    )
                break
            lives = new_lives
        if verbose:
            print(f"Seed {seed}: Reward: {total_reward}. Steps: {step_count}")
        env.close()
        total_rewards.append(total_reward)
        episode_lengths.append(step_count)

    rewards = np.array(total_rewards)
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    if verbose:
        print(f"Average Reward over {len(seeds)} seeds: {avg_reward}")
        print(f"Standard Deviation: {std_reward}")

    if return_lists:
        return total_rewards, episode_lengths
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
        n_seeds=args.n_seeds,
        deterministic=args.deterministic,
        verbose=True,
    )
