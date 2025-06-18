import os
from stable_baselines3 import A2C, PPO
from components.environment import make_atari_env
from components.wrappers import EncoderWrapper
from components.encoder import RuleBasedEncoder
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack

import yaml
import os
import argparse


def eval(args):
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    game_name = config["environment"]["game_name"]
    seed = config["environment"]["seed"]
    model_name = config["model"]["name"]
    model_path = "./weights"

    agent_mappings = {
        "player+ball": {
            "encoding_method": "paddle+ball",
            "n_features": 5,
            "name": model_name + "_rb_player_ball",
        },
        "player+ball+bricks": {
            "encoding_method": "bricks+paddle+ball",
            "n_features": 113,
            "name": model_name + "_rb_player_ball_bricks",
        },
        "transformer": {
            "encoding_method": "transformer",
            "n_features": -1,
            "name": model_name + "_rb_transformer",
        },
        "cnn": {
            "encoding_method": "cnn",
            "n_features": -1,
            "name": model_name + "_cnn",
        },
        "rule_based": {
            "encoding_method": "rule_based",
            "n_features": -1,
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
            "greyscale": True if args.agent == "transformer" else False,
            "screen_size": -1,
            "clip_reward": False,
            "terminal_on_life_loss": False,
            "max_pool": False,
        }

    env = make_atari_env(game_name, n_envs=1, seed=seed, wrapper_kwargs=wrapper_kwargs)
    if args.agent == "cnn":
        # Stack frames to encode temporal information
        env = VecFrameStack(env, n_stack=4)
    else:
        if args.agent == "transformer":
            # Stack frames to encode temporal information
            env = VecFrameStack(env, n_stack=2)
        env = EncoderWrapper(env, encoder, n_features)

    # Load model
    model = PPO.load(
        os.path.join(model_path, agent_mappings[args.agent]["name"]),
        env=env,
        seed=seed,
        custom_objects={
            "observation_space": env.observation_space,
            "action_space": env.action_space,
        },
    )

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=10, render=False
    )
    print(f"Mean reward: {mean_reward} +/- {std_reward}")
    env.close()


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
            "rule_based",
        ],
        required=True,
        help="The agent type to evaluate.",
    )
    args = parser.parse_args()
    eval(args)
