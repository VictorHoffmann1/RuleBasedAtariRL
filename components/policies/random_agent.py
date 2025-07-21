import numpy as np


class RandomAgent:
    def __init__(self, num_actions, seed=0):
        self.seed = seed
        np.random.seed(seed)
        self.num_actions = num_actions

    def set_random_seed(self, seed: int):
        """
        Set random seed for reproducibility.
        """
        self.seed = seed
        np.random.seed(seed)

    def predict(self, features, deterministic: bool = True):
        """
        Perform an action based on the observation.
        Vectorized implementation using numpy.
        """
        actions = np.random.randint(low=0, high=self.num_actions, size=len(features))
        return actions, {}
