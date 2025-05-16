import gym
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation, RecordVideo
import random
import numpy as np

def create_environment(
        game_name="ALE/SpaceInvaders-v5",
        render_mode="rgb_array",
        record_video=False,
        num_envs=1,
        video_dir="videos"
        ):

    env = gym.vector.make(game_name, 
                          render_mode=render_mode,
                          repeat_action_probability=0.1, 
                          num_envs=num_envs) if num_envs > 1 \
        else gym.make(game_name, render_mode=render_mode)

    if record_video:
        env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda x: True)

    return env

def make_env(game_name="ALE/SpaceInvaders-v5",
            render_mode="rgb_array",
            seed=42):
    def _init():
        env = create_environment(
            game_name=game_name,
            render_mode=render_mode
        )
        env.reset(seed=seed + np.random.randint(0, 1000))  # Seed each worker differently
        return env
    return _init
