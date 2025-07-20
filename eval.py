from components.agent_mappings import get_agent_mapping
from components.utils import create_env, load_model
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecMonitor,
    is_vecenv_wrapped,
)
import yaml
import os
import numpy as np
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def eval(
    model=None,
    env=None,
    agent: str = "unknown",
    model_extension: str = "eval",
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    callback=None,
    reward_threshold: float = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
):
    if env is None or model is None:
        # Load configuration
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        n_envs = 1 if env is None else env.num_envs
        agent_mapping = get_agent_mapping(
            agent, config["environment"]["game_name"], model_extension
        )

    if env is None:
        env = create_env(config, agent_mapping, n_envs, seed=0, train=False)

    if model is None:
        model_path = "./weights"
        model = load_model(
            env, agent_mapping, os.path.join(model_path, agent_mapping["name"]), seed=0
        )

    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = (
        is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]
    )

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array(
        [(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int"
    )

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)

    # Check if FIRE action is available (action 1)
    has_fire_action = False
    try:
        action_meanings = env.unwrapped.get_action_meanings()
        if len(action_meanings) > 1 and action_meanings[1] == "FIRE":
            has_fire_action = True
    except (AttributeError, IndexError):
        # If the environment doesn't support get_action_meanings or doesn't have action 1
        has_fire_action = False
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    # Check if this is a life loss (not true episode end) and FIRE action is available
                    is_life_loss = False
                    is_true_episode_end = False

                    if is_monitor_wrapped:
                        if "episode" in info.keys():
                            # This is a true episode end
                            is_true_episode_end = True
                        else:
                            # This is likely a life loss, not a true episode end
                            is_life_loss = True
                    else:
                        # For non-monitor wrapped envs, we treat all dones as true episode ends
                        # unless we can detect otherwise from the info
                        is_true_episode_end = True

                    # If agent loses a life and FIRE action is available, perform FIRE action
                    if is_life_loss and has_fire_action:
                        # Create actions array with FIRE action for the specific environment
                        fire_actions = np.zeros(n_envs, dtype=int)
                        fire_actions[i] = 1  # FIRE action
                        fire_obs, fire_rewards, fire_dones, fire_infos = env.step(
                            fire_actions
                        )
                        # Update observations with the fire step
                        new_observations = fire_obs
                        # Update rewards and lengths with fire step
                        current_rewards[i] += fire_rewards[i]
                        current_lengths[i] += 1

                    # Handle episode completion tracking
                    if is_true_episode_end:
                        if is_monitor_wrapped and "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                        else:
                            episode_rewards.append(current_rewards[i])
                            episode_lengths.append(current_lengths[i])
                            episode_counts[i] += 1
                        current_rewards[i] = 0
                        current_lengths[i] = 0

        observations = new_observations

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, (
            f"Mean reward below threshold: {mean_reward:.2f} < {reward_threshold:.2f}"
        )
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="The agent type to evaluate.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="The model type to evaluate.",
    )
    parser.add_argument(
        "--deterministic",
        type=bool,
        default=True,
        help="Run the evaluation in deterministic mode.",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=10,
        help="Number of random seeds to use for evaluation.",
    )
    args = parser.parse_args()
    eval(
        None,
        agent=args.agent,
        model_extension=args.model,
        n_seeds=args.n_seeds,
        deterministic=args.deterministic,
        verbose=True,
    )
