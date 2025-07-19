from stable_baselines3 import PPO
from components.environment import make_atari_env
from components.wrappers import EncoderWrapper
from components.agent_mappings import get_agent_mapping
from components.schedulers import linear_scheduler, exponential_scheduler
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback
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

def create_env(agent_mapping, n_envs, game_name, seed):
    """Create environment with given parameters"""
    wrapper_kwargs = agent_mapping.get("wrapper_kwargs", {})
    wrapper_kwargs["frame_skip"] = 4 if "NoFrameskip" in game_name else 1
    wrapper_kwargs["noop_max"] = 30 if "v4" in game_name else 0
    wrapper_kwargs["terminal_on_life_loss"] = True if "v4" in game_name else False
    wrapper_kwargs["time_limit"] = 10000 if "NoFrameskip" in game_name else 2000
    env = make_atari_env(
        game_name,
        n_envs=n_envs,
        seed=seed,
        wrapper_kwargs=agent_mapping["wrapper_kwargs"],

        env_kwargs={
            "obs_type": "rgb",
            "frameskip": 5,
            "mode": None,
            "difficulty": None,
            "repeat_action_probability": 0.25,
            "full_action_space": True,
            "render_mode": None,
        } if "v5" in game_name else {}
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
    for environment in env.envs:
        check_env(environment, warn=True)
    return env

def create_model(env, config, agent_mapping, seed, log_dir):
    # Initialize the model
    model_params = config["model"]

    model = PPO(
        agent_mapping["policy"],
        env,
        verbose=2,
        policy_kwargs={"n_features": agent_mapping["n_features"]}
        if agent_mapping["use_feature_kwargs"]
        else {},
        learning_rate=exponential_scheduler(
            model_params["learning_rate"],
            1e-5,
            #model_params["learning_rate"]
            #* (1 - config["training"]["num_steps"] / 1e7),
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
            model_params["clip_range"] * 0.1,
        )
        if model_params["scheduler"]
        else model_params["clip_range"],
        max_grad_norm=model_params["max_grad_norm"],
        tensorboard_log=log_dir,
        seed=seed,
    )

    return model

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
    # Get agent mappings configuration
    agent_mapping = get_agent_mapping(args.agent, config, n_envs, game_name)

    # Set up TensorBoard log directory
    log_dir = "./logs/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    weights_dir = "./weights/"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    env = create_env(agent_mapping, n_envs, game_name, seed)
    eval_env = create_env(agent_mapping, 1, game_name, seed + 1000)

    model = create_model(env, config, agent_mapping, seed + 500, log_dir)

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=max(100000 // n_envs, 1),
        verbose=1,
        n_eval_episodes=10,
        deterministic=True,
    )

    model.learn(
        total_timesteps=config["training"]["num_steps"],
        tb_log_name=agent_mapping["name"],
        #callback=eval_callback,
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
        default="player+ball",
        required=True,
        help="The agent type to test.",
    )
    args = parser.parse_args()
    train(args)
