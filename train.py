from stable_baselines3 import A2C, PPO
from components.environment import make_atari_env
from components.wrappers import EncoderWrapper
from components.encoder import RuleBasedEncoder
from components.transformer_encoder import CustomTransformerPolicy
from components.deep_sets_encoder import CustomDeepSetPolicy
from stable_baselines3.common.vec_env import VecFrameStack
import yaml
import os
import numpy as np
import argparse


def train(args):
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    n_envs = config["environment"]["number"]
    game_name = config["environment"]["game_name"]
    seed = config["environment"]["seed"]
    model_name = config["model"]["name"]

    agent_mappings = {
        "player+ball": {
            "encoding_method": "paddle+ball",
            "n_features": 5,
            "name": model_name + "_rb_player_ball",
            "policy": "MlpPolicy",
            "n_stack": None,
        },
        "player+ball+bricks": {
            "encoding_method": "bricks+paddle+ball",
            "n_features": 113,
            "name": model_name + "_rb_player_ball_bricks",
            "policy": "MlpPolicy",
            "n_stack": None,
        },
        "transformer": {
            "encoding_method": "transformer",
            "n_features": 8,
            "name": model_name + "_rb_transformer",
            "policy": CustomTransformerPolicy,
            "n_stack": 2,  # Stack frames for temporal encoding
        },
        "deep_sets": {
            "encoding_method": "transformer",
            "n_features": 8,
            "name": model_name + "_rb_deep_sets",
            "policy": CustomDeepSetPolicy,
            "n_stack": 2,  # Stack frames for temporal encoding
        },
        "cnn": {
            "encoding_method": "cnn",
            "n_features": -1,
            "name": model_name + "_cnn",
            "policy": "CnnPolicy",
            "n_stack": 4,  # Stack frames for CNN
        },
    }

    n_features = agent_mappings[args.agent]["n_features"]
    config["encoder"]["encoding_method"] = agent_mappings[args.agent]["encoding_method"]
    encoder = RuleBasedEncoder(**config["encoder"])

    if args.agent == "cnn":
        wrapper_kwargs = {}
    else:
        wrapper_kwargs = {
            "greyscale": True if args.agent in ["transformer", "deep_sets"] else False,
            "screen_size": -1,
            "max_pool": False,
        }

    env = make_atari_env(
        game_name, n_envs=n_envs, seed=seed, wrapper_kwargs=wrapper_kwargs
    )

    if args.agent in ["transformer", "deep_sets", "cnn"]:
        # Stack frames to encode temporal information
        env = VecFrameStack(env, n_stack=agent_mappings[args.agent]["n_stack"])
    if args.agent != "cnn":
        env = EncoderWrapper(env, encoder, n_features)

    # Set up TensorBoard log directory
    log_dir = "./logs/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    weights_dir = "./weights/"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    model_params = config["model"]

    if model_name == "A2C":
        model = A2C(
            agent_mappings[args.agent]["policy"],
            env,
            verbose=2,
            learning_rate=model_params["learning_rate"],
            n_steps=model_params["n_steps"],
            gamma=model_params["gamma"],
            gae_lambda=model_params["gae_lambda"],
            ent_coef=model_params["ent_coef"],
            vf_coef=model_params["vf_coef"],
            tensorboard_log=log_dir,
        )

    elif model_name == "PPO":
        model = PPO(
            agent_mappings[args.agent]["policy"],
            env,
            verbose=2,
            learning_rate= exponential_schedule(model_params["lr_start"], model_params["lr_end"]) \
                if model_params["scheduler"] else model_params["learning_rate"],
            batch_size=model_params["ppo_batch_size"],
            n_epochs=model_params["n_epochs"],
            n_steps=model_params["n_steps"],
            gamma=model_params["gamma"],
            gae_lambda=model_params["gae_lambda"],
            ent_coef=model_params["ent_coef"],
            vf_coef=model_params["vf_coef"],
            tensorboard_log=log_dir,
        )

    model.learn(
        total_timesteps=config["training"]["num_episodes"],
        tb_log_name=agent_mappings[args.agent]["name"],
    )

    # Save model
    model.save(os.path.join(weights_dir, agent_mappings[args.agent]["name"]))
    env.close()

# Exponential LR schedule from 1e-3 to 1e-5
def exponential_schedule(initial_value: float, final_value: float):
    log_initial = np.log(initial_value)
    log_final = np.log(final_value)

    def func(progress_remaining: float) -> float:
        # Convert progress_remaining (1 → 0) into fraction of progress (0 → 1)
        frac = 1.0 - progress_remaining
        log_lr = log_initial + frac * (log_final - log_initial)
        return float(np.exp(log_lr))
    
    return func


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
            "deep_sets",
            "cnn",
        ],
        required=True,
        help="The agent type to test.",
    )
    args = parser.parse_args()
    train(args)
