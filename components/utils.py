from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from components.environment import make_oc_atari_env
from components.wrappers import OCAtariEncoderWrapper
from stable_baselines3 import PPO
from components.policies.naive_agent import NaiveAgent
from components.policies.random_agent import RandomAgent


def create_env(config, agent_mapping, n_envs, seed, train=True):
    """Create environment with given parameters"""
    game_name = config["environment"]["game_name"]
    wrapper_kwargs = {
        "clip_reward": True if train else False,
        "noop_max": 30 if "v4" in game_name and train else 0,
        "terminal_on_life_loss": True if "v4" in game_name and train else False,
    }

    if agent_mapping["policy"] == "CnnPolicy":
        wrapper_kwargs["frame_skip"] = 4 if "v4" in game_name else 5
        env_kwargs = (
            {
                "frameskip": 1,
                "repeat_action_probability": 0.25,
                "full_action_space": True,
            }
            if "v5" in game_name
            else {}
        )
        env = make_atari_env(
            game_name,
            n_envs=n_envs,
            seed=seed,
            env_kwargs=env_kwargs,
            wrapper_kwargs=wrapper_kwargs,
        )
        # Stack frames to encode temporal information
        env = VecFrameStack(env, n_stack=agent_mapping["n_stack"])
        if train:
            env = VecTransposeImage(env)
    else:
        wrapper_kwargs["frame_skip"] = 4 if "NoFrameskip" in game_name else 1
        wrapper_kwargs["max_pool"] = False
        wrapper_kwargs["time_limit"] = 10000 if "NoFrameskip" in game_name else 2000
        env_kwargs = {
            "mode": "ram",
            "hud": False,
            "obs_mode": "ori",
        }
        if "v5" in game_name:
            env_kwargs["frameskip"] = 5
            env_kwargs["repeat_action_probability"] = 0.25
            env_kwargs["full_action_space"] = True
        env = make_oc_atari_env(
            game_name,
            n_envs=n_envs,
            seed=seed,
            env_kwargs=env_kwargs,
            wrapper_kwargs=wrapper_kwargs,
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
    return env


def load_model(env, agent_mapping, path=None, seed=0):
    if agent_mapping["policy"] == "Naive":
        model = NaiveAgent()
    elif agent_mapping["policy"] == "Random":
        model = RandomAgent(num_actions=env.action_space.n, seed=seed)
    else:
        if path is None:
            raise ValueError("Model path must be provided for non-naive agents.")
        model = PPO.load(
            path,
            env=env,
            seed=seed,
            custom_objects={
                "observation_space": env.observation_space,
                "action_space": env.action_space,
            },
        )

    return model
