from typing import Any, SupportsFloat, Tuple, Dict
from stable_baselines3.common.vec_env import VecEnvWrapper

import gym
import numpy as np
from gym import spaces

try:
    import cv2

    cv2.ocl.setUseOpenCL(False)
except ImportError:
    cv2 = None  # type: ignore[assignment]

AtariResetReturn = Tuple[np.ndarray, Dict[str, Any]]
AtariStepReturn = Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]


class EncoderWrapper(VecEnvWrapper):
    """
    Wrapper to encode the observations using a rule-based encoder.
    :param venv: Environment to wrap
    :param encoder: Encoder class/function to apply to each observation
    :param n_features: Number of features in the encoded observation
    """

    # TODO: Add support for custom transfomer encoder
    def __init__(self, venv, encoder, n_features):
        super().__init__(venv)
        self.encoder = encoder
        if (
            "discovery" in self.encoder.method
            or "object_vectors" in self.encoder.method
        ):
            shape = (self.encoder.max_objects, n_features)
        else:
            shape = (n_features,)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=shape, dtype=np.float32
        )

    def reset(self):
        obs = self.venv.reset()
        return self.encoder(obs)

    def step_wait(self):
        obs, rewards, terminated, infos = self.venv.step_wait()
        encoded_obs = self.encoder(obs)
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
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = np.zeros(0)
        info: dict = {}
        for _ in range(noops):
            step_result = self.env.step(self.noop_action)
            # Handle different gym versions
            if len(step_result) == 5:
                obs, _, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, _, done, info = step_result
            if done:
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
        self.env.reset(**kwargs)
        # Step 1
        step_result = self.env.step(1)
        if len(step_result) == 5:
            obs, _, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            obs, _, done, _ = step_result
        if done:
            obs, _ = self.env.reset(**kwargs)
        # Step 2
        step_result = self.env.step(2)
        if len(step_result) == 5:
            obs, _, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            obs, _, done, _ = step_result
        if done:
            obs, _ = self.env.reset(**kwargs)
        return obs, {}


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
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            obs, reward, done, info = step_result
            terminated = done
            truncated = False
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> AtariResetReturn:
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            step_result = self.env.step(0)
            if len(step_result) == 5:
                obs, _, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, _, done, info = step_result
            if done:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every ``skip``-th frame (frameskipping)
    and optionally return the max between the two last frames.

    :param env: Environment to wrap
    :param skip: Number of ``skip``-th frame
        The same action will be taken ``skip`` times.
    :param max_pool: If True, return the max over the last two frames. If False, return the last frame only.
    """

    def __init__(self, env: gym.Env, skip: int = 4, max_pool: bool = True) -> None:
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        assert env.observation_space.dtype is not None, (
            "No dtype specified for the observation space"
        )
        assert env.observation_space.shape is not None, (
            "No shape defined for the observation space"
        )
        self._obs_buffer = np.zeros(
            (2, *env.observation_space.shape), dtype=env.observation_space.dtype
        )
        self._skip = skip
        self._max_pool = max_pool

    def step(self, action: int) -> AtariStepReturn:
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations (if enabled).

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        for i in range(self._skip):
            step_result = self.env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
                terminated = done
                truncated = False
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += float(reward)
            if done:
                break
        if self._max_pool:
            max_frame = self._obs_buffer.max(axis=0)
            return max_frame, total_reward, terminated, truncated, info
        else:
            return self._obs_buffer[1], total_reward, terminated, truncated, info


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


class WarpFrame(gym.ObservationWrapper):
    """
    Convert to grayscale and warp frames to 84x84 (default)
    as done in the Nature paper and later work.

    :param env: Environment to wrap
    :param width: New frame width
    :param height: New frame height
    """

    def __init__(
        self, env: gym.Env, width: int = 84, height: int = 84, greyscale: bool = True
    ) -> None:
        super().__init__(env)
        self.width = width
        self.height = height
        self.greyscale = greyscale
        rgb_channel = 1 if self.greyscale else 3

        assert isinstance(env.observation_space, spaces.Box), (
            f"Expected Box space, got {env.observation_space}"
        )
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, rgb_channel),
            dtype=env.observation_space.dtype,  # type: ignore[arg-type]
        )

    def observation(self, frame) -> np.ndarray:
        """
        returns the current observation from a frame

        :param frame: environment frame
        :return: the observation
        """
        if type(frame) == tuple:
            # Handle the case where the environment returns obs, info
            frame = frame[0]
        assert cv2 is not None, (
            "OpenCV is not installed, you can do `pip install opencv-python`"
        )
        if self.greyscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if self.width != frame.shape[1] or self.height != frame.shape[0]:
            frame = cv2.resize(
                frame, (self.width, self.height), interpolation=cv2.INTER_AREA
            )
        return frame[:, :, None] if self.greyscale else frame


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
        frame_skip: int = 4,
        max_pool: bool = True,
        screen_size: int = 84,
        terminal_on_life_loss: bool = True,
        clip_reward: bool = True,
        action_repeat_probability: float = 0.0,
        greyscale: bool = True,
    ) -> None:
        if action_repeat_probability > 0.0:
            env = StickyActionEnv(env, action_repeat_probability)
        if noop_max > 0:
            env = NoopResetEnv(env, noop_max=noop_max)
        # frame_skip=1 is the same as no frame-skip (action repeat)
        if frame_skip > 1:
            env = MaxAndSkipEnv(env, skip=frame_skip, max_pool=max_pool)
        if terminal_on_life_loss:
            env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():  # type: ignore[attr-defined]
            env = FireResetEnv(env)
        if screen_size > 0:
            width, height = screen_size, screen_size
        else:  # use original_size
            height, width = env.observation_space.shape[0:2]
        env = WarpFrame(env, width=width, height=height, greyscale=greyscale)
        if clip_reward:
            env = ClipRewardEnv(env)

        super().__init__(env)
