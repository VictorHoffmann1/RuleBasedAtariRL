from stable_baselines3 import PPO
from components.environment import make_oc_atari_env
from components.wrappers import OCAtariEncoderWrapper
from components.agent_mappings import get_agent_mapping
from components.schedulers import linear_scheduler, exponential_scheduler, get_lr
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
import yaml
import os
import argparse
import optuna
import numpy as np


def optuna_search(args):
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    game_name = config["environment"]["game_name"]
    n_envs = config["environment"]["number"]

    # Get agent mappings configuration
    agent_mapping = get_agent_mapping(
        args.agent, game_name, model_extension="optuna"
    )

    # Early stopping configuration
    early_stop_enabled = getattr(args, "early_stop", True)
    confidence_threshold = getattr(args, "confidence_threshold", 0.9)  # 90% confidence

    def calculate_probability_exceed_best(current_rewards, best_value):
        """
        Calculate the probability that the current trial will exceed the best known value
        using a simple statistical approach based on current performance.
        """
        if len(current_rewards) < 2:
            return 1.0  # Not enough data, assume it could be good

        current_mean = np.mean(current_rewards)
        current_std = np.std(current_rewards, ddof=1)

        if current_std == 0:
            # No variance, use mean comparison
            return 1.0 if current_mean > best_value else 0.0

        # Estimate the distribution and calculate probability
        # Using normal approximation (central limit theorem)
        z_score = (best_value - current_mean) / (
            current_std / np.sqrt(len(current_rewards))
        )

        # Probability that mean > best_value is 1 - CDF(z_score)
        # Using approximation for normal CDF
        if z_score > 6:
            return 0.0
        elif z_score < -6:
            return 1.0
        else:
            # Normal CDF approximation
            prob_exceed = 0.5 * (1 + np.tanh(z_score * np.sqrt(2 / np.pi)))
            return 1.0 - prob_exceed

    def objective(trial, study):
        env_seeds = list(range(3))  # Use multiple seeds for robustness
        model_seeds = list(range(10, 13))  # Different seeds for model

        if args.agent == "cnn":
            env = make_atari_env(
                game_name,
                n_envs=n_envs,
                seed=0,
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
                seed=0,
                env_kwargs=oc_atari_kwargs,
            )
        if agent_mapping["encoder"]:
            env = OCAtariEncoderWrapper(
                env,
                config["encoder"]["max_objects"],
                method=agent_mapping["method"],
                num_envs=n_envs,
                speed_scale=config["encoder"]["speed_scale"],
                use_rgb=config["encoder"]["use_rgb"],
                use_category=config["encoder"]["use_category"],
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
        #ent_coef = trial.suggest_float("ent_coef", 1e-6, 1e-2, log=True)
        clip_range = trial.suggest_float("clip_range", 0.4, 0.4)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1.5e-3)
        batch_size = trial.suggest_categorical("batch_size", [128, 256])
        n_epochs = trial.suggest_categorical("n_epochs", [5])
        gae_lambda = trial.suggest_float("gae_lambda", 0.95, 0.95)

        model = PPO(
            agent_mapping["policy"],
            env,
            verbose=0,
            learning_rate=exponential_scheduler(
                learning_rate,
                get_lr("exponential", learning_rate, args.training_steps, 1e-5, 1e7),
            ),
            batch_size=batch_size,
            n_epochs=n_epochs,
            n_steps=n_steps,
            gamma=config["model"]["gamma"],
            gae_lambda=gae_lambda,
            ent_coef=0,
            vf_coef=config["model"]["vf_coef"],
            clip_range=linear_scheduler(
                clip_range,
                get_lr("linear", clip_range, args.training_steps, clip_range * 0.1, 1e7)
            ),
            max_grad_norm=config["model"]["max_grad_norm"],
            tensorboard_log=log_dir,
            seed=0,
        )

        # stop_train_callback = StopTrainingOnNoModelImprovement(
        #    max_no_improvement_evals=3, min_evals=0, verbose=1
        # )
        # eval_callback = EvalCallback(
        #    env,
        #    eval_freq=max(100000 // n_envs, 1),
        #    callback_after_eval=stop_train_callback,
        #    verbose=1,
        # )

        rewards = []

        # Get current best value for early stopping
        best_value = float("-inf") if len(study.trials) == 1 else study.best_value

        for i, (env_seed, model_seed) in enumerate(zip(env_seeds, model_seeds)):
            model.set_random_seed(model_seed)
            model.policy.reinitialize_weights()
            model.policy.optimizer = model.policy.optimizer.__class__(
                model.policy.parameters(), **model.policy.optimizer.defaults
            )
            env.seed(env_seed)
            env.reset()

            model.learn(
                total_timesteps=args.training_steps,
                tb_log_name=agent_mapping["name"],
                # callback=eval_callback,
            )

            # Evaluate the model
            reward, _ = evaluate_policy(
                model=model,
                env=env,
                n_eval_episodes=10,
            )
            print(f"Trial {trial.number}, Seed {env_seed}, Reward: {reward}")

            rewards.append(reward)

            # Early stopping check
            if (
                early_stop_enabled and len(study.trials) > 0
            ):  # Only if we have previous trials to compare
                prob_exceed = calculate_probability_exceed_best(rewards, best_value)
                if prob_exceed < (1 - confidence_threshold):
                    print(
                        f"Early stopping after {i + 1} seeds (low probability of improvement)"
                    )
                    break

        # Use percentiles for more accurate quartile calculation
        q25 = np.percentile(rewards, 25)
        q75 = np.percentile(rewards, 75)
        inter_quartile_rewards = [r for r in rewards if q25 <= r <= q75]
        inter_quartile_mean = np.mean(inter_quartile_rewards)

        return inter_quartile_mean

    # Run the optimization
    study = optuna.create_study(direction="maximize")

    # Create a wrapper for the objective function that has access to the study
    def objective_with_study(trial):
        return objective(trial, study)

    study.optimize(objective_with_study, n_trials=40, n_jobs=1)

    # Best hyperparameters
    print("-" * 30)
    print("HYPERPARAMETER TUNING FINSIHED")
    print("-" * 30)
    print("BEST TRIAL:")
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
    parser.add_argument(
        "--training_steps",
        type=int,
        default=1000000,
        help="Total number of training steps for each trial.",
    )
    parser.add_argument(
        "--early_stop",
        action="store_true",
        help="Enable early stopping based on statistical confidence.",
    )
    parser.add_argument(
        "--min_seeds",
        type=int,
        default=3,
        help="Minimum number of seeds to evaluate before considering early stopping.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.9,
        help="Confidence threshold for early stopping (0.9 = 90% confidence).",
    )
    args = parser.parse_args()
    optuna_search(args)
