from stable_baselines3 import A2C, PPO
from components.environment import make_atari_env
from components.wrappers import EncoderWrapper
from components.encoders.breakout_encoder import BreakoutEncoder
from components.encoders.object_discovery_encoder import ObjectDiscoveryEncoder
from components.transformer_encoder import CustomTransformerPolicy
from components.deep_sets_encoder import CustomDeepSetPolicy
from components.schedulers import linear_scheduler
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

    rb_encoder = {
        "BreakoutNoFrameskip-v4": BreakoutEncoder,
        # "PongNoFrameskip-v4": PongEncoder,
    }

    agent_mappings = {
        "player+ball": {
            "encoder": rb_encoder[game_name](
                encoding_method="paddle+ball",
                speed_scale=config["encoder"]["speed_scale"],
                num_envs=n_envs,
            ),
            "n_features": 5 if "Breakout" in game_name else 6,
            "name": model_name + "_rb_player_ball",
            "policy": "MlpPolicy",
            "n_stack": None,
        },
        "player+ball+bricks": {
            "encoder": rb_encoder[game_name](
                encoding_method="bricks+paddle+ball",
                speed_scale=config["encoder"]["speed_scale"],
                num_envs=n_envs,
            ),
            "n_features": 113,
            "name": model_name + "_rb_player_ball_bricks",
            "policy": "MlpPolicy",
            "n_stack": None,
        },
        "transformer": {
            "encoder": ObjectDiscoveryEncoder(
                speed_scale=config["encoder"]["speed_scale"],
                num_envs=n_envs,
                max_objects=config["encoder"]["max_objects"],
            ),
            "n_features": 8,
            "name": model_name + "_rb_transformer",
            "policy": CustomTransformerPolicy,
            "n_stack": 2,  # Stack frames for temporal encoding
        },
        "deep_sets": {
            "encoder": ObjectDiscoveryEncoder(
                speed_scale=config["encoder"]["speed_scale"],
                num_envs=n_envs,
                max_objects=config["encoder"]["max_objects"],
            ),
            "n_features": 8,
            "name": model_name + "_rb_deep_sets",
            "policy": CustomDeepSetPolicy,
            "n_stack": 2,  # Stack frames for temporal encoding
        },
        "cnn": {
            "encoder": None,  # CNN does not require a custom encoder
            "n_features": -1,
            "name": model_name + "_cnn",
            "policy": "CnnPolicy",
            "n_stack": 4,  # Stack frames for CNN
        },
    }

    if args.agent == "cnn":
        wrapper_kwargs = {}
    else:
        wrapper_kwargs = {
            "screen_size": -1,
            "max_pool": False,
        }

    env = make_atari_env(
        game_name, n_envs=n_envs, seed=seed, wrapper_kwargs=wrapper_kwargs
    )

    if agent_mappings[args.agent]["n_stack"] is not None:
        # Stack frames to encode temporal information
        env = VecFrameStack(env, n_stack=agent_mappings[args.agent]["n_stack"])
    if agent_mappings[args.agent]["encoder"] is not None:
        env = EncoderWrapper(
            env,
            agent_mappings[args.agent]["encoder"],
            agent_mappings[args.agent]["n_features"],
        )

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
            max_grad_norm=model_params["max_grad_norm"],
            tensorboard_log=log_dir,
            seed=seed,
        )

    elif model_name == "PPO":
        model = PPO(
            agent_mappings[args.agent]["policy"],
            env,
            verbose=2,
            learning_rate=linear_scheduler(
                model_params["learning_rate"],
                model_params["learning_rate"]
                * (1 - config["training"]["num_steps"] / 1e7),
            )
            if model_params["scheduler"]
            else model_params["learning_rate"],
            batch_size=model_params["ppo_batch_size"],
            n_epochs=model_params["n_epochs"],
            n_steps=model_params["n_steps"],
            gamma=model_params["gamma"],
            gae_lambda=model_params["gae_lambda"],
            ent_coef=model_params["ent_coef"],
            vf_coef=model_params["vf_coef"],
            clip_range=linear_scheduler(
                model_params["clip_range"],
                model_params["clip_range"]
                * (1 - config["training"]["num_steps"] / 1e7),
            )
            if model_params["scheduler"]
            else model_params["clip_range"],
            max_grad_norm=model_params["max_grad_norm"],
            tensorboard_log=log_dir,
            seed=seed,
        )

    model.learn(
        total_timesteps=config["training"]["num_steps"],
        tb_log_name=agent_mappings[args.agent]["name"],
    )

    # Save model
    model.save(os.path.join(weights_dir, agent_mappings[args.agent]["name"]))
    env.close()


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
