import os
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from components.environment import make_atari_env
import yaml


def train():
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    n_envs = config["environment"]["number"]
    game_name = config["environment"]["game_name"]
    seed = config["environment"]["seed"]
    model_name = config["model"]["name"]

    # Create vectorized Atari environment
    env = make_atari_env(game_name, n_envs=n_envs, seed=seed)

    # Stack frames (to capture motion)
    env = VecFrameStack(env, n_stack=4)

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
            "CnnPolicy",
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
            "CnnPolicy",
            env,
            verbose=2,
            learning_rate=model_params["learning_rate"],
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
        tb_log_name=model_name + "_breakout_cnn",
    )

    # Save model
    model.save(os.path.join(weights_dir, model_name + "_breakout_cnn"))
    env.close()


if __name__ == "__main__":
    train()
