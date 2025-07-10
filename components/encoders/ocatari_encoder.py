import numpy as np
import cv2
from typing import List, Optional
import gymnasium as gym
from ocatari.core import OCAtari
from ocatari.vision.game_objects import GameObject
import warnings


class OCAtariEncoder:
    """Rule-based encoder for Atari games with batched environment support."""

    def __init__(
        self,
        max_objects: int = 64,
        speed_scale: float = 10.0,
        num_envs: int = 1,
        method: str = "discovery",
        use_rgb: bool = False,
        use_category: bool = False,
    ):
        self.num_envs = num_envs
        self.speed_scale = speed_scale
        self.img_width = 160  # Width of the Atari screen
        self.img_height = 210  # Height of the Atari screen
        self.n_features = 6 if method == "discovery" else 5
        if use_rgb:
            self.n_features += 3
        if use_category:
            self.n_features += 3
        self.max_objects = max_objects
        self.method = method
        self.use_rgb = use_rgb
        self.use_category = use_category

        self.normed_width = RunningNormalizer()
        self.normed_height = RunningNormalizer()
        self.normed_dx = RunningNormalizer()
        self.normed_dy = RunningNormalizer()

        assert self.method in ["discovery", "expert"], (
            f"Method must be 'discovery' or 'expert', got {self.method}."
        )

    def __call__(self, envs) -> np.ndarray:
        """Encodes a batch of frames into feature spaces by return paddle position and ball position + velocity.
        Args:
            states: Batch of input frames [num_envs, H, W, C]
        Returns:
            np.ndarray: Batch of encoded features [num_envs, feature_dim]
        """

        batch_features = []

        for env in envs.envs:
            objects = self.get_ocatari_objects(env)

            if self.method == "discovery":
                features = np.zeros(
                    (self.max_objects, self.n_features)
                )  # Initialize features for each env
                idx = 0
                for object in objects:
                    if object.category != "NoObject":
                        if idx >= self.max_objects:
                            warnings.warn(
                                f"More objects than max_objects ({self.max_objects}) in environment. Truncating.",
                                UserWarning,
                            )
                            break
                        object_vector = np.array(
                            [
                                self.normalize(object.center[0], self.img_width, True),
                                self.normalize(object.center[1], self.img_height, True),
                                self.normed_dx(object.dx),
                                self.normed_dy(object.dy),
                                self.normed_width(object.w),
                                self.normed_height(object.h),
                            ]
                        )
                        if self.use_rgb:
                            rgb = object.rgb
                            rgb_vector = np.array(
                                [
                                    self.normalize(rgb[0], 255, True),
                                    self.normalize(rgb[1], 255, True),
                                    self.normalize(rgb[2], 255, True),
                                ]
                            )
                            object_vector = np.concatenate((object_vector, rgb_vector))
                        if self.use_category:
                            category_vector = np.zeros(3)
                            if object.category == "Player":
                                category_vector[0] = 1.0
                            elif object.category == "Ball":
                                category_vector[1] = 1.0
                            else:
                                category_vector[2] = 1.0
                            object_vector = np.concatenate(
                                (object_vector, category_vector)
                            )
                        features[idx] = object_vector
                        idx += 1

            elif self.method == "expert":
                features = np.zeros(
                    (self.n_features)
                )  # paddle_x, ball_x, ball_y, ball_dx, ball_dy
                features[1:3] = -2.0  # Initialize ball position to -2.0
                for object in objects:
                    if object.category == "Player":
                        features[0] = self.normalize(
                            object.x, self.img_width, True
                        )
                    elif object.category == "Ball":
                        features[1] = self.normalize(
                            object.x, self.img_width, True
                        )
                        features[2] = self.normalize(
                            object.y, self.img_height, True
                        )
                        features[3] = self.normed_dx(object.dx)
                        features[4] = self.normed_dy(object.dy)

            batch_features.append(features)

        return np.stack(batch_features)  # [num_envs, feature_dim]

    @staticmethod
    def get_ocatari_objects(env: gym.Env):
        """
        Get the objects from the underlying OCAtari environment through the wrapper chain.

        :param env: The wrapped environment
        :return: The objects from the OCAtari environment
        """
        # Traverse through wrappers to find the OCAtari instance
        current_env = env
        while hasattr(current_env, "env"):
            if isinstance(current_env, OCAtari):
                return current_env.objects
            current_env = current_env.env

        # Check if the current env is OCAtari
        if isinstance(current_env, OCAtari):
            return current_env.objects

        # If we can't find OCAtari, raise an error
        raise ValueError("No OCAtari environment found in the wrapper chain")

    @staticmethod
    def normalize(value, scale, centering=False):
        """
        Normalize a value to the range [-1, 1] based on a scale.

        :param value: The value to normalize
        :param scale: The scale for normalization
        :param centering: If True, center the value around 0 before scaling
        :return: Normalized value in the range [-1, 1]
        """
        if centering:
            return np.clip(2 * (value / scale) - 1, -1, 1) if scale != 0 else 0
        else:
            return np.clip(value / scale, -1, 1) if scale != 0 else 0


class RunningNormalizer:
    """A simple running normalizer for a feature."""

    def __init__(self, epsilon: float = 1e-8):
        self.sum = 0.0
        self.count = 0
        self.sum_squared = 0.0
        self.epsilon = epsilon  # Small value to avoid division by zero

    def __call__(self, x: float):
        """Normalize a single value."""
        self.sum += x
        self.count += 1
        self.sum_squared += x**2
        return np.clip((x - self.mean()) / (self.std() + self.epsilon), -10, 10)

    def mean(self):
        """Return the mean of the feature."""
        return self.sum / self.count if self.count > 0 else 0.0

    def std(self):
        """Return the standard deviation of the feature."""
        if self.count > 1:
            return np.sqrt(
                (self.sum_squared - (self.sum**2) / self.count) / (self.count - 1)
            )
        return 0.0
