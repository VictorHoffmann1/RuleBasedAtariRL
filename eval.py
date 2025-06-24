import os
from stable_baselines3 import A2C, PPO
from components.environment import make_atari_env
from components.wrappers import EncoderWrapper
from components.encoder import RuleBasedEncoder
from components.naive_agent import NaiveAgent
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

import yaml
import os
import numpy as np
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def eval(args):
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    game_name = config["environment"]["game_name"]
    model_name = config["model"]["name"]
    model_path = "./weights"

    agent_mappings = {
        "player+ball": {
            "encoding_method": "paddle+ball",
            "n_features": 5,
            "name": model_name + "_rb_player_ball_" + args.model,
        },
        "player+ball+bricks": {
            "encoding_method": "bricks+paddle+ball",
            "n_features": 113,
            "name": model_name + "_rb_player_ball_bricks_" + args.model,
        },
        "transformer": {
            "encoding_method": "transformer",
            "n_features": 8,
            "name": model_name + "_rb_transformer_" + args.model,
        },
        "deep_sets": {
            "encoding_method": "transformer",
            "n_features": 8,
            "name": model_name + "_rb_deep_sets_" + args.model,
            "n_stack": 2,  # Stack frames for temporal encoding
        },
        "cnn": {
            "encoding_method": "cnn",
            "n_features": -1,
            "name": model_name + "_cnn_" + args.model,
        },
        "rule_based": {
            "encoding_method": "paddle+ball",
            "n_features": 5,
            "name": None,
        },
    }

    n_features = agent_mappings[args.agent]["n_features"]
    config["encoder"]["encoding_method"] = agent_mappings[args.agent]["encoding_method"]

    encoder = RuleBasedEncoder(**config["encoder"])

    if args.agent == "cnn":
        wrapper_kwargs = {
            "clip_reward": False,
            "terminal_on_life_loss": False,
        }
    else:
        wrapper_kwargs = {
            "greyscale": True if args.agent in ["transformer", "deep_sets"] else False,
            "screen_size": -1,
            "clip_reward": False,
            "terminal_on_life_loss": False,
            "max_pool": False,
        }

    seeds = list(range(10))  # Use a range of seeds for evaluation
    total_rewards = []
    for seed in seeds:
        env = make_atari_env(
            game_name, n_envs=1, seed=seed, wrapper_kwargs=wrapper_kwargs
        )
        if args.agent == "cnn":
            # Stack frames to encode temporal information
            env = VecFrameStack(env, n_stack=4)
            env = VecTransposeImage(env)  # Transpose for CNN input
        else:
            if args.agent in ["transformer", "deep_sets"]:
                # Stack frames to encode temporal information
                env = VecFrameStack(env, n_stack=2)
            env = EncoderWrapper(env, encoder, n_features)

        if args.agent == "rule_based":
            model = NaiveAgent()
        else:
            # Load model
            if model_name == "A2C":
                model = A2C.load(
                    os.path.join(model_path, agent_mappings[args.agent]["name"]),
                    env=env,
                    seed=seed,
                    custom_objects={
                        "observation_space": env.observation_space,
                        "action_space": env.action_space,
                    },
                )
            elif model_name == "PPO":
                model = PPO.load(
                    os.path.join(model_path, agent_mappings[args.agent]["name"]),
                    env=env,
                    seed=seed,
                    custom_objects={
                        "observation_space": env.observation_space,
                        "action_space": env.action_space,
                    },
                )
            else:
                raise ValueError(f"Model {model_name} not implemented.")

        obs = env.reset()
        done = [False]

        total_reward = 0
        step_count = 0
        # --- Track per-life stats ---
        per_life_reward = 0
        per_life_step = 0
        # Get initial lives from info dict after first step
        actions, _ = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(actions)
        if isinstance(info, list):
            info = info[0]
        lives = info.get("lives", None)
        # Reset env to first obs after getting lives
        obs = env.reset()
        done = [False]
        while not done[0]:
            actions, _ = model.predict(obs, deterministic=False)
            # Force Fire action if the model doesn't predict it
            if per_life_step > 300 and per_life_reward == 0.0:
                actions[0] = 1
            obs, reward, done, info = env.step(actions)
            step_count += 1
            total_reward += reward[0]
            per_life_reward += reward[0]
            per_life_step += 1
            if isinstance(info, list):
                info = info[0]
            new_lives = info.get("lives", lives)
            if new_lives < lives:
                per_life_reward = 0
                per_life_step = 0
            lives = new_lives
        print(f"Seed {seed}: Reward: {total_reward}. Steps: {step_count}")
        env.close()
        total_rewards.append(total_reward)

    print(f"Average Reward over {len(seeds)} seeds: {sum(total_rewards) / len(seeds)}")
    print(f"Total Rewards: {total_rewards}")
    print(f"Standard Deviation: {np.std(total_rewards)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument(
        "--agent",
        type=str,
        choices=[
            "player+ball",
            "player+ball+bricks",
            "transformer",
            "cnn",
            "deep_sets",
            "rule_based",
        ],
        required=True,
        help="The agent type to evaluate.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="The model type to evaluate.",
    )
    args = parser.parse_args()
    eval(args)
