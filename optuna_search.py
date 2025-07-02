from stable_baselines3 import A2C, PPO
from components.environment import make_oc_atari_env
from components.wrappers import OCAtariEncoderWrapper
from components.agent_mappings import get_agent_mapping
from components.schedulers import linear_scheduler, exponential_scheduler
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.callbacks import  EvalCallback, StopTrainingOnNoModelImprovement
from eval import eval
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

    # Get agent mappings configuration
    agent_mapping = get_agent_mapping(args.agent, 
                                      game_name, 
                                      model_name, 
                                      model_extension="optuna")

    def objective(trial):
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
                speed_scale=config["encoder"]["speed_scale"],
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
        gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)

        stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=0, verbose=1)
        eval_callback = EvalCallback(env, eval_freq=max(100000 // n_envs, 1), callback_after_eval=stop_train_callback, verbose=1)

        if model_name == "A2C":
            model = A2C(
                agent_mapping["policy"],
                env,
                verbose=0,
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
                agent_mapping["policy"],
                env,
                verbose=0,
                learning_rate=linear_scheduler(
                    learning_rate,
                    learning_rate * (1 - config["training"]["num_steps"] / 1e7),
                ),
                batch_size=batch_size,
                n_epochs=n_epochs,
                n_steps=n_steps,
                gamma=config["model"]["gamma"],
                gae_lambda=gae_lambda,
                ent_coef=ent_coef,
                vf_coef=config["model"]["vf_coef"],
                clip_range=linear_scheduler(
                    clip_range, clip_range * (1 - config["training"]["num_steps"] / 1e7)
                ),
                max_grad_norm=config["model"]["max_grad_norm"],
                tensorboard_log=log_dir,
                seed=seed,
            )

        model.learn(
            total_timesteps=config["training"]["num_steps"],
            tb_log_name=agent_mapping["name"],
            callback=eval_callback,
        )

        # Evaluate the model
        mean_reward, _ = eval(
            model=model,
            agent=args.agent,
            deterministic=True,
            n_seeds=20,
            verbose=False,
        )

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
        required=True,
        help="The agent type to test.",
    )
    args = parser.parse_args()
    optuna_search(args)
