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
        use_rgb: bool = False,
    ):
        self.num_envs = num_envs
        self.speed_scale = speed_scale
        self.img_width = 160  # Width of the Atari screen
        self.img_height = 210  # Height of the Atari screen
        self.n_features = 9 if use_rgb else 6
        self.max_objects = max_objects
        self.use_rgb = use_rgb

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
                    if self.use_rgb:
                        rgb = object.rgb
                        rgb_vector = np.array(
                            self.normalize(rgb[0], 255, "[0,1]"),
                            self.normalize(rgb[1], 255, "[0,1]"),
                            self.normalize(rgb[2], 255, "[0,1]"),
                        )
                    object_vector = np.array(
                        [
                            self.normalize(object.x, self.img_width, "[-1,1]"),
                            self.normalize(object.y, self.img_height, "[-1,1]"),
                            self.normalize(
                                object.dx, self.img_width / self.speed_scale,
                                 "[0,1]"
                            ),
                            self.normalize(
                                object.dy, self.img_height / self.speed_scale,
                                 "[0,1]"
                            ),
                            self.normalize(object.w, self.img_width, "[0,1]"),
                            self.normalize(object.h, self.img_height, "[0,1]"),
                        ]
                    )
                    features[idx, :6] = object_vector
                    if self.use_rgb:
                        features[idx, 6:] = rgb_vector
                    idx += 1

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
    def normalize(value, scale, range_type="[0,1]"):
        """
        Normalize a value to the range [-1, 1] based on a scale.

        :param value: The value to normalize
        :param scale: The scale for normalization
        :return: Normalized value in the range [-1, 1]
        """
        if range_type == "[0,1]":
            # Normalize to [0, 1]
            return value / scale if scale != 0 else 0
        elif range_type == "[-1,1]":
            return 2 * (value / scale) - 1 if scale != 0 else 0
        else:
            raise ValueError(f"Unsupported range type: {range_type}. Use '[0,1]' or '[-1,1]'.")
