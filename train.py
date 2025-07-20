import argparse
import datetime
import multiprocessing as mp
import os

import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env

from components.agent_mappings import get_agent_mapping
from components.schedulers import get_lr
from components.utils import create_env


# Optimize CPU performance - dynamic thread allocation
def set_optimal_threads():
    cpu_cores = mp.cpu_count()
    # For high-core systems, allow more threads but prevent oversubscription
    # Rule: max threads = cpu_cores // 2 for vectorized environments
    optimal_threads = max(1, min(cpu_cores // 2, cpu_cores - 2))

    # Override if environment variable is set
    if "TORCH_NUM_THREADS" in os.environ:
        optimal_threads = int(os.environ["TORCH_NUM_THREADS"])

    print(f"Setting {optimal_threads} threads for {cpu_cores} CPU cores")
    return optimal_threads


def get_optimal_env_count():
    """Determine optimal number of environments based on CPU cores"""
    cpu_cores = mp.cpu_count()
    # Use config value or scale with CPU cores, whichever is smaller
    optimal_envs = max(4, cpu_cores // 2)
    print(f"CPU cores: {cpu_cores}, Using {optimal_envs} environments")
    return optimal_envs


optimal_threads = set_optimal_threads()
torch.set_num_threads(optimal_threads)
os.environ["OMP_NUM_THREADS"] = str(optimal_threads)
os.environ["MKL_NUM_THREADS"] = str(optimal_threads)


def train(args):
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    n_envs = (
        get_optimal_env_count()
        if config["environment"]["use_optimal_cores"]
        else config["environment"]["number"]
    )
    game_name = config["environment"]["game_name"]
    seed = config["environment"]["seed"]
    n_features = 6
    if config["encoder"]["use_rgb"]:
        n_features += 3
    if config["encoder"]["use_category"]:
        n_features += 3

    # Get agent mappings configuration
    agent_mapping = get_agent_mapping(args.agent, game_name)

    # Create training environment
    env = create_env(config, agent_mapping, n_envs, seed, train=True)

    # Create evaluation environment with different seed
    eval_env = create_env(config, agent_mapping, n_envs, seed + 1000, train=False)

    # Check if the environments are valid
    for environment in env.envs:
        check_env(environment)
    for environment in eval_env.envs:
        check_env(environment)

    # Set up TensorBoard log directory
    log_dir = "./logs/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    weights_dir = "./weights/"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    # Initialize the model
    model_params = config["model"]

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=max(100000 // n_envs, 1),
        verbose=1,
        n_eval_episodes=10,
        deterministic=True,
    )

    print("Training configuration:")
    print(f"  - Environments: {n_envs}")
    print(f"  - Batch size: {model_params['ppo_batch_size']}")
    print(f"  - Steps per rollout: {model_params['n_steps']}")
    print(f"  - Total rollout size: {n_envs * model_params['n_steps']}")
    print(f"  - PyTorch threads: {torch.get_num_threads()}")

    model = PPO(
        agent_mapping["policy"],
        env,
        verbose=2,
        learning_rate=get_lr(
            model_params["scheduler"],
            model_params["learning_rate"],
            config["training"]["num_steps"],
            0 if model_params["scheduler"] == "linear" else 1e-5,
            1e7,
        ),
        batch_size=model_params["ppo_batch_size"],
        n_epochs=model_params["n_epochs"],
        n_steps=model_params["n_steps"],
        gamma=model_params["gamma"],
        gae_lambda=model_params["gae_lambda"],
        ent_coef=model_params["ent_coef"],
        vf_coef=model_params["vf_coef"],
        clip_range=get_lr(
            "linear",
            model_params["clip_range"],
            config["training"]["num_steps"],
            0.1 * model_params["clip_range"],
            1e7,
        ),
        max_grad_norm=model_params["max_grad_norm"],
        tensorboard_log=log_dir,
        seed=seed + 500,  # Different seed for model
        policy_kwargs={"n_features": n_features}
        if agent_mapping["use_feature_kwargs"]
        else {},
    )

    model.learn(
        total_timesteps=config["training"]["num_steps"],
        # callback=eval_callback,
        tb_log_name=agent_mapping["name"],
        progress_bar=True,
    )

    # Save model with unique filename to avoid overwriting
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{agent_mapping['name']}_{timestamp}"
    model.save(os.path.join(weights_dir, model_filename))
    env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Rule-Based Encoder")
    parser.add_argument(
        "--agent",
        type=str,
        default="deepsets",
        required=True,
        help="The agent type to test.",
    )
    args = parser.parse_args()
    train(args)
