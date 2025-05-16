import gym
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation, RecordVideo
import random

def create_environment(
        game_name="ALE/SpaceInvaders-v5",
        render_mode="rgb_array",
        record_video=False,
        noop_max=30,
        video_dir="videos"
        ):

    env = gym.make(game_name, render_mode=render_mode)
    env = NoopResetEnv(env, noop_max=noop_max)  # Random number of no-op actions on reset

    if record_video:
        env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda x: True)

    return env


class NoopResetEnv(gym.Wrapper):
    """Wrapper to perform a random number of no-op actions on reset."""	
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0  # Typically '0' is the NOOP in Atari

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = random.randint(1, self.noop_max)
        # Fire first
        obs, _, _, _, _ = self.env.step(1)
        for _ in range(noops):
            obs, _, done, _, info = self.env.step(self.noop_action)
            if done:
                obs, info = self.env.reset(**kwargs)
        return obs, info