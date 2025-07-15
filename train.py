from stable_baselines3 import A2C, PPO
from components.environment import make_oc_atari_env
from components.wrappers import OCAtariEncoderWrapper
from components.agent_mappings import get_agent_mapping
from components.schedulers import linear_scheduler, exponential_scheduler
from components.vec_normalizer import VecNormalize
from components.callbacks import CustomEvalCallback
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.env_checker import check_env
import yaml
import os
import argparse
import datetime
import torch
import multiprocessing as mp


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


def create_env(args, config, agent_mapping, n_envs, game_name, seed):
    """Create environment with given parameters"""
    if args.agent == "cnn":
        env = make_atari_env(
            game_name,
            n_envs=n_envs,
            seed=seed,
        )
        # Stack frames to encode temporal information
        env = VecFrameStack(env, n_stack=agent_mapping["n_stack"])
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
            seed=seed,
            env_kwargs=oc_atari_kwargs,
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
        # env = VecNormalize(
        #    env,
        #    norm_obs=False,
        #    norm_reward=True,
        # )
    return env


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
    env = create_env(args, config, agent_mapping, n_envs, game_name, seed)

    # Create evaluation environment with different seed
    eval_env = create_env(args, config, agent_mapping, n_envs, game_name, seed + 1000)

    # Check if the environment is valid
    for environment in env.envs:
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

    eval_callback = CustomEvalCallback(
        eval_env,
        eval_freq=max(10000 // n_envs, 1),
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
        learning_rate=exponential_scheduler(
            model_params["learning_rate"],
            1e-5,
            # model_params["learning_rate"] * (1 - config["training"]["num_steps"] / 1e7),
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
            model_params["clip_range"] * (1 - config["training"]["num_steps"] / 1e7),
        )
        if model_params["scheduler"]
        else model_params["clip_range"],
        max_grad_norm=model_params["max_grad_norm"],
        tensorboard_log=log_dir,
        seed=seed + 500,  # Different seed for model
        policy_kwargs={"n_features": n_features}
        if agent_mapping["use_feature_kwargs"]
        else {},
    )

    model.learn(
        total_timesteps=config["training"]["num_steps"],
        callback=eval_callback,
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
