from typing import Any, SupportsFloat, Tuple, Dict
from stable_baselines3.common.vec_env import VecEnvWrapper
from components.encoders.ocatari_encoder import OCAtariEncoder
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import cv2

    cv2.ocl.setUseOpenCL(False)
except ImportError:
    cv2 = None  # type: ignore[assignment]

AtariResetReturn = Tuple[np.ndarray, Dict[str, Any]]
AtariStepReturn = Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]


class OCAtariEncoderWrapper(VecEnvWrapper):
    """
    Wrapper to encode the observations using a rule-based encoder.
    :param venv: Environment to wrap
    :param encoder: Encoder class/function to apply to each observation
    :param n_features: Number of features in the encoded observation
    """

    def __init__(self, venv, max_objects, num_envs, speed_scale=10.0, use_rgb=False, use_category=False):
        super().__init__(venv)
        self.encoder = OCAtariEncoder(
            max_objects=max_objects,
            speed_scale=speed_scale,
            num_envs=num_envs,
            use_rgb=use_rgb,
            use_category=use_category
        )
        shape = (max_objects, self.encoder.n_features)  # Each object has 6 features: x, y, dx, dy, w, h
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=shape, dtype=np.float32
        )

    def reset(self):
        _ = self.venv.reset()
        return self.encoder(self.venv)

    def step_wait(self):
        _, rewards, terminated, infos = self.venv.step_wait()
        encoded_obs = self.encoder(self.venv)

        # Process terminal observations in infos to match the encoded observation space
        for i, info in enumerate(infos):
            if "terminal_observation" in info:
                # The terminal observation should also be encoded to match our observation space
                # Since we can't encode a single terminal observation without the environment state,
                # we'll use the current encoded observation as a reasonable approximation
                info["terminal_observation"] = encoded_obs[i]

        return encoded_obs, rewards, terminated, infos


class StickyActionEnv(gym.Wrapper):
    """
    Sticky action.

    Paper: https://arxiv.org/abs/1709.06009
    Official implementation: https://github.com/mgbellemare/Arcade-Learning-Environment

    :param env: Environment to wrap
    :param action_repeat_probability: Probability of repeating the last action
    """

    def __init__(self, env: gym.Env, action_repeat_probability: float) -> None:
        super().__init__(env)
        self.action_repeat_probability = action_repeat_probability
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"  # type: ignore[attr-defined]

    def reset(self, **kwargs) -> AtariResetReturn:
        self._sticky_action = 0  # NOOP
        return self.env.reset(**kwargs)

    def step(self, action: int) -> AtariStepReturn:
        if self.np_random.random() >= self.action_repeat_probability:
            self._sticky_action = action
        return self.env.step(self._sticky_action)


class NoopResetEnv(gym.Wrapper):
    """
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.

    :param env: Environment to wrap
    :param noop_max: Maximum value of no-ops to run
    """

    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"  # type: ignore[attr-defined]

    def reset(self, **kwargs) -> AtariResetReturn:
        obs, info = self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        # obs and info are already set from the reset call above
        for _ in range(noops):
            step_result = self.env.step(self.noop_action)
            obs, _, terminated, truncated, info = step_result
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class FireResetEnv(gym.Wrapper):
    """
    Take action on reset for environments that are fixed until firing.

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"  # type: ignore[attr-defined]
        assert len(env.unwrapped.get_action_meanings()) >= 3  # type: ignore[attr-defined]

    def reset(self, **kwargs) -> AtariResetReturn:
        obs, info = self.env.reset(**kwargs)
        # Step 1
        step_result = self.env.step(1)
        obs, _, terminated, truncated, info = step_result
        done = terminated or truncated
        if done:
            obs, info = self.env.reset(**kwargs)
        # Step 2
        step_result = self.env.step(2)
        obs, _, terminated, truncated, info = step_result
        done = terminated or truncated
        if done:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class EpisodicLifeEnv(gym.Wrapper):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action: int) -> AtariStepReturn:
        step_result = self.env.step(action)
        obs, reward, terminated, truncated, info = step_result
        self.was_real_done = terminated or truncated
        lives = info["lives"]
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> AtariResetReturn:
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            step_result = self.env.step(0)
            obs, _, terminated, truncated, info = step_result
            done = terminated or truncated
            if done:
                obs, info = self.env.reset(**kwargs)
        self.lives = info["lives"]  # type: ignore[attr-defined]
        return obs, info


class ClipRewardEnv(gym.RewardWrapper):
    """
    Clip the reward to {+1, 0, -1} by its sign.

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def reward(self, reward: SupportsFloat) -> float:
        """
        Bin reward to {+1, 0, -1} by its sign.

        :param reward:
        :return:
        """
        return np.sign(float(reward))


class AtariWrapper(gym.Wrapper):
    """
    Atari 2600 preprocessings

    Specifically:

    * Noop reset: obtain initial state by taking random number of no-ops on reset.
    * Frame skipping: 4 by default
    * Max-pooling: most recent two observations
    * Termination signal when a life is lost.
    * Resize to a square image: 84x84 by default
    * Grayscale observation
    * Clip reward to {-1, 0, 1}
    * Sticky actions: disabled by default

    See https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
    for a visual explanation.

    .. warning::
        Use this wrapper only with Atari v4 without frame skip: ``env_id = "*NoFrameskip-v4"``.

    :param env: Environment to wrap
    :param noop_max: Max number of no-ops
    :param frame_skip: Frequency at which the agent experiences the game.
        This correspond to repeating the action ``frame_skip`` times.
    :param screen_size: Resize Atari frame
    :param terminal_on_life_loss: If True, then step() returns done=True whenever a life is lost.
    :param clip_reward: If True (default), the reward is clip to {-1, 0, 1} depending on its sign.
    :param action_repeat_probability: Probability of repeating the last action
    :param warp_frame: If True (default), the frame is resized to 84x84 and converted to grayscale.
    """

    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 30,
        terminal_on_life_loss: bool = True,
        clip_reward: bool = True,
        action_repeat_probability: float = 0.0,
    ) -> None:
        if action_repeat_probability > 0.0:
            env = StickyActionEnv(env, action_repeat_probability)
        if noop_max > 0:
            env = NoopResetEnv(env, noop_max=noop_max)
        if terminal_on_life_loss:
            env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():  # type: ignore[attr-defined]
            env = FireResetEnv(env)
        if clip_reward:
            env = ClipRewardEnv(env)

        super().__init__(env)
