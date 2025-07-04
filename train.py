from stable_baselines3 import A2C, PPO
from components.environment import make_atari_env
from components.wrappers import EncoderWrapper
from components.agent_mappings import get_agent_mapping
from components.schedulers import linear_scheduler, exponential_scheduler
from stable_baselines3.common.vec_env import VecFrameStack
import yaml
import os
import argparse


def train(args):
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    n_envs = config["environment"]["number"]
    game_name = config["environment"]["game_name"]
    seed = config["environment"]["seed"]
    model_name = config["model"]["name"]

    # Get agent mappings configuration
    agent_mapping = get_agent_mapping(args.agent, config, n_envs, game_name, model_name)

    env = make_atari_env(
        game_name,
        n_envs=n_envs,
        seed=seed,
        wrapper_kwargs=agent_mapping["wrapper_kwargs"],
    )

    if agent_mapping["n_stack"] is not None:
        # Stack frames to encode temporal information
        env = VecFrameStack(env, n_stack=agent_mapping["n_stack"])
    if agent_mapping["encoder"] is not None:
        env = EncoderWrapper(
            env,
            agent_mapping["encoder"],
            agent_mapping["n_features"],
        )

    # Set up TensorBoard log directory
    log_dir = "./logs/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    weights_dir = "./weights/"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    # Initialize the model
    model_params = config["model"]

    if model_name == "A2C":
        model = A2C(
            agent_mapping["policy"],
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
            agent_mapping["policy"],
            env,
            verbose=2,
            policy_kwargs={"n_features": agent_mapping["n_features"]}
            if agent_mapping["use_feature_kwargs"]
            else {},
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
        tb_log_name=agent_mapping["name"],
    )

    # Save model
    model.save(os.path.join(weights_dir, agent_mapping["name"]))
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Rule-Based Encoder")
    parser.add_argument(
        "--agent",
        type=str,
        default="player+ball",
        required=True,
        help="The agent type to test.",
    )
    args = parser.parse_args()
    train(args)
