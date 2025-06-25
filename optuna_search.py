from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from components.environment import make_atari_env
from components.wrappers import EncoderWrapper
from components.encoders.breakout_encoder import BreakoutEncoder
from components.encoders.object_discovery_encoder import ObjectDiscoveryEncoder
from components.transformer_encoder import CustomTransformerPolicy
from components.deep_sets_encoder import CustomDeepSetPolicy
from components.schedulers import linear_scheduler
import yaml
import os
import argparse
import optuna


def optuna_search(args):
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    game_name = config["environment"]["game_name"]
    seed = config["environment"]["seed"]
    model_name = config["model"]["name"]
    n_envs = config["environment"]["number"]

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
            "n_features": 5,
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

    def objective(trial):
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

        # Sample hyperparameters
        n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024, 2048])
        ent_coef = trial.suggest_float("ent_coef", 1e-4, 0.1, log=True)
        clip_range = trial.suggest_float("clip_range", 0.05, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        n_epochs = trial.suggest_categorical("n_epochs", [4, 5, 8, 10])

        if model_name == "A2C":
            model = A2C(
                agent_mappings[args.agent]["policy"],
                env,
                verbose=2,
                learning_rate=learning_rate["learning_rate"],
                n_steps=n_steps,
                gamma=config["model"]["gamma"],
                gae_lambda=config["model"]["gae_lambda"],
                ent_coef=ent_coef,
                vf_coef=config["model"]["vf_coef"],
                max_grad_norm=config["model"]["max_grad_norm"],
                tensorboard_log=log_dir,
                seed=seed,
            )

        elif model_name == "PPO":
            model = PPO(
                agent_mappings[args.agent]["policy"],
                env,
                verbose=1,
                learning_rate=linear_scheduler(
                    learning_rate,
                    learning_rate * (1 - config["training"]["num_steps"] / 1e7),
                ),
                batch_size=batch_size,
                n_epochs=n_epochs,
                n_steps=n_steps,
                gamma=config["model"]["gamma"],
                gae_lambda=config["model"]["gae_lambda"],
                ent_coef=ent_coef,
                vf_coef=config["model"]["vf_coef"],
                clip_range=clip_range,
                max_grad_norm=config["model"]["max_grad_norm"],
                tensorboard_log=log_dir,
                seed=seed,
            )

        model.learn(
            total_timesteps=config["training"]["num_steps"],
            tb_log_name=agent_mappings[args.agent]["name"],
        )

        # Evaluate the model
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)

        return mean_reward

    # Run the optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40, n_jobs=1)

    # Best hyperparameters
    print("Best trial:")
    print(study.best_trial)

    # Best hyperparameters
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")


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
    optuna_search(args)
