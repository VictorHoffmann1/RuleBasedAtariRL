import pickle
import warnings

import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper


class VecNormalize(VecEnvWrapper):
    """
    A moving average, normalizing wrapper for vectorized environment.

    It is pickleable which will save moving averages and configuration parameters.
    The wrapped environment `venv` is not saved, and must be restored manually with
    `set_venv` after being unpickled.

    :param venv: (VecEnv) the vectorized environment to wrap
    :param training: (bool) Whether to update or not the moving average
    :param norm_obs: (bool) Whether to normalize observation or not (default: True)
    :param norm_reward: (bool) Whether to normalize rewards or not (default: True)
    :param clip_obs: (float) Max absolute value for observation
    :param clip_reward: (float) Max value absolute for discounted reward
    :param gamma: (float) discount factor
    :param epsilon: (float) To avoid division by zero
    """

    def __init__(
        self,
        venv,
        training=True,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
        epsilon=1e-8,
    ):
        VecEnvWrapper.__init__(self, venv)
        self.obs_rms = RunningMeanStd(shape=(self.observation_space.shape[-1],))
        self.ret_rms = RunningMeanStd(shape=())
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        # Returns: discounted rewards
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = training
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.old_obs = None
        self.old_rews = None

    def __getstate__(self):
        """
        Gets state for pickling.

        Excludes self.venv, as in general VecEnv's may not be pickleable."""
        state = self.__dict__.copy()
        # these attributes are not pickleable
        del state["venv"]
        del state["class_attributes"]
        # these attributes depend on the above and so we would prefer not to pickle
        del state["ret"]
        return state

    def __setstate__(self, state):
        """
        Restores pickled state.

        User must call set_venv() after unpickling before using.

        :param state: (dict)"""
        self.__dict__.update(state)
        assert "venv" not in state
        self.venv = None

    def set_venv(self, venv):
        """
        Sets the vector environment to wrap to venv.

        Also sets attributes derived from this such as `num_env`.

        :param venv: (VecEnv)
        """
        if self.venv is not None:
            raise ValueError(
                "Trying to set venv of already initialized VecNormalize wrapper."
            )
        VecEnvWrapper.__init__(self, venv)
        if self.obs_rms.mean.shape != self.observation_space.shape:
            raise ValueError("venv is incompatible with current statistics.")
        self.ret = np.zeros(self.num_envs)

    def step_wait(self):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step_wait()
        self.old_obs = obs
        self.old_rews = rews

        if self.training:
            obs_reshaped = obs.reshape(-1, obs.shape[-1])
            self.obs_rms.update(obs_reshaped)
        obs = self.normalize_obs(obs)

        if self.training:
            self._update_reward(rews)
        rews = self.normalize_reward(rews)

        self.ret[news] = 0
        return obs, rews, news, infos

    def _update_reward(self, reward: np.ndarray) -> None:
        """Update reward normalization statistics."""
        self.ret = self.ret * self.gamma + reward
        self.ret_rms.update(self.ret)

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize observations using this VecNormalize's observations statistics.
        Calling this method does not update statistics.
        """
        if self.norm_obs:
            obs = np.clip(
                (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
                -self.clip_obs,
                self.clip_obs,
            )
        return obs

    def normalize_reward(self, reward: np.ndarray) -> np.ndarray:
        """
        Normalize rewards using this VecNormalize's rewards statistics.
        Calling this method does not update statistics.
        """
        if self.norm_reward:
            reward = np.clip(
                reward / np.sqrt(self.ret_rms.var + self.epsilon),
                -self.clip_reward,
                self.clip_reward,
            )
        return reward

    def get_original_obs(self) -> np.ndarray:
        """
        Returns an unnormalized version of the observations from the most recent
        step or reset.
        """
        return self.old_obs.copy()

    def get_original_reward(self) -> np.ndarray:
        """
        Returns an unnormalized version of the rewards from the most recent step.
        """
        return self.old_rews.copy()

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        self.old_obs = obs
        self.ret = np.zeros(self.num_envs)
        if self.training:
            self._update_reward(self.ret)
        return self.normalize_obs(obs)

    @staticmethod
    def load(load_path, venv):
        """
        Loads a saved VecNormalize object.

        :param load_path: the path to load from.
        :param venv: the VecEnv to wrap.
        :return: (VecNormalize)
        """
        with open(load_path, "rb") as file_handler:
            vec_normalize = pickle.load(file_handler)
        vec_normalize.set_venv(venv)
        return vec_normalize

    def save(self, save_path):
        with open(save_path, "wb") as file_handler:
            pickle.dump(self, file_handler)

    def save_running_average(self, path):
        """
        :param path: (str) path to log dir

        .. deprecated:: 2.9.0
            This function will be removed in a future version
        """
        warnings.warn(
            "Usage of `save_running_average` is deprecated. Please "
            "use `save` or pickle instead.",
            DeprecationWarning,
        )
        for rms, name in zip([self.obs_rms, self.ret_rms], ["obs_rms", "ret_rms"]):
            with open("{}/{}.pkl".format(path, name), "wb") as file_handler:
                pickle.dump(rms, file_handler)

    def load_running_average(self, path):
        """
        :param path: (str) path to log dir

        .. deprecated:: 2.9.0
            This function will be removed in a future version
        """
        warnings.warn(
            "Usage of `load_running_average` is deprecated. Please "
            "use `load` or pickle instead.",
            DeprecationWarning,
        )
        for name in ["obs_rms", "ret_rms"]:
            with open("{}/{}.pkl".format(path, name), "rb") as file_handler:
                setattr(self, name, pickle.load(file_handler))


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: tuple[int, ...] = ()):
        """
        Calculates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean.copy()
        new_object.var = self.var.copy()
        new_object.count = float(self.count)
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.

        :param other: The other object to combine with.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
