import os
from stable_baselines3 import A2C, PPO
from environment import make_atari_env
from wrappers import EncoderWrapper
from encoder import RuleBasedEncoder
from stable_baselines3.common.evaluation import evaluate_policy

import yaml
import os

def eval():

    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    game_name = config["environment"]["game_name"]
    seed = config["environment"]["seed"]
    n_features = config["model"]["n_features"]
    model_path = "./weights"

    encoder = RuleBasedEncoder(**config["encoder"])

    rule_based_kwargs = {
        "greyscale": False,
        "screen_size": -1,
        "clip_reward": False,
    }

    env = make_atari_env(game_name, n_envs=1, seed=seed, wrapper_kwargs=rule_based_kwargs)
    env = EncoderWrapper(env, encoder, n_features)

    # Load model
    model = PPO.load(os.path.join(model_path, "ppo_breakout_rb"), env=env, seed=seed,
                    custom_objects={'observation_space': env.observation_space, 'action_space': env.action_space})

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # Recording video example

    env.close()

if __name__ == "__main__":
    eval()