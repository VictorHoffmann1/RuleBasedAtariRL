import warnings

import gymnasium as gym
import numpy as np
from ocatari.core import OCAtari


class OCAtariEncoder:
    """Rule-based encoder for Atari games with batched environment support."""

    IMG_WIDTH = 160  # Default Atari image width
    IMG_HEIGHT = 210  # Default Atari image height

    def __init__(
        self,
        max_objects: int = 64,
        speed_scale: float = 10.0,
        num_envs: int = 1,
        method: str = "discovery",
        use_rgb: bool = False,
        use_category: bool = False,
        use_events: bool = False,
    ):
        self.num_envs = num_envs
        self.speed_scale = speed_scale
        self.n_features = 6 if method == "discovery" else 5
        if use_rgb and method == "discovery":
            self.n_features += 3
        if use_category and method == "discovery":
            self.n_features += 3
        if use_events and method == "expert":
            self.n_features += 4
            self.left_wall = Object(4, 113.5, 8, 165, 0, 0)
            self.right_wall = Object(156, 113.5, 8, 165, 0, 0)
            self.top_wall = Object(80, 24, 160, 14, 0, 0)
        self.max_objects = max_objects
        self.method = method
        self.use_rgb = use_rgb
        self.use_events = use_events
        self.use_category = use_category

        # Add caching for object extraction to avoid repeated wrapper traversal
        self._env_ocatari_cache = {}

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
                                self.normalize(object.center[0], self.IMG_WIDTH, True),
                                self.normalize(object.center[1], self.IMG_HEIGHT, True),
                                self.normalize(object.dx, self.speed_scale, False),
                                self.normalize(object.dy, self.speed_scale, False),
                                self.normalize(object.w, self.IMG_WIDTH, False),
                                self.normalize(object.h, self.IMG_HEIGHT, False),
                            ]
                        )
                        if self.use_rgb:
                            rgb = object.rgb
                            rgb_vector = np.array(
                                [
                                    self.normalize(rgb[0], 255, False),
                                    self.normalize(rgb[1], 255, False),
                                    self.normalize(rgb[2], 255, False),
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
                )  # paddle_x, ball_x, ball_y, ball_dx, ball_dy and optional events
                # Events: ball -> block, ball -> player, ball -> wall and ball -> lost
                features[1:3] = -2.0  # Initialize ball position to -2.0
                player_found, ball_found = False, False
                for object in objects:
                    # In the objects list, the first object is always the player
                    # and the second is always the ball.
                    if object.category == "Player":
                        player_object = object
                        features[0] = self.normalize(
                            player_object.center[0], self.IMG_WIDTH, True
                        )
                        player_found = True
                    elif object.category == "Ball":
                        ball_object = object
                        features[1] = self.normalize(
                            ball_object.center[0], self.IMG_WIDTH, True
                        )
                        features[2] = self.normalize(
                            ball_object.center[1], self.IMG_HEIGHT, True
                        )
                        features[3] = self.normalize(
                            ball_object.dx, self.speed_scale, False
                        )
                        features[4] = self.normalize(
                            ball_object.dy, self.speed_scale, False
                        )
                        ball_found = True
                        if self.use_events:
                            features[8] = 1.0
                            if player_found:
                                # Check if the ball is colliding with the player
                                if self.is_collision(ball_object, player_object):
                                    features[6] = 1.0
                            # Check if the ball is colliding with the walls
                            if any(
                                [
                                    self.is_collision(ball_object, self.left_wall),
                                    self.is_collision(ball_object, self.right_wall),
                                    self.is_collision(ball_object, self.top_wall),
                                ]
                            ):
                                features[7] = 1.0
                    elif object.category == "Block" and ball_found:
                        # Check if the ball is colliding with a block
                        if self.is_collision(ball_object, object):
                            features[5] = 1.0
                            break  # There can only be one block collision at a time
                    if player_found and ball_found:
                        break

            batch_features.append(features)

        return np.stack(batch_features)  # [num_envs, feature_dim]

    def get_ocatari_objects(self, env: gym.Env):
        """
        Get the objects from the underlying OCAtari environment through the wrapper chain.
        Uses caching to avoid repeated wrapper traversal.

        :param env: The wrapped environment
        :return: The objects from the OCAtari environment
        """
        env_id = id(env)

        # Check cache first
        if env_id in self._env_ocatari_cache:
            return self._env_ocatari_cache[env_id].objects

        # Traverse through wrappers to find the OCAtari instance
        current_env = env
        while hasattr(current_env, "env"):
            if isinstance(current_env, OCAtari):
                self._env_ocatari_cache[env_id] = current_env
                return current_env.objects
            current_env = current_env.env

        # Check if the current env is OCAtari
        if isinstance(current_env, OCAtari):
            self._env_ocatari_cache[env_id] = current_env
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

    @staticmethod
    def is_collision(obj_1, obj_2):
        pos_1 = np.array(obj_1.center)
        pos_2 = np.array(obj_2.center)
        speed_1 = np.array([obj_1.dx, obj_1.dy])
        speed_2 = np.array([obj_2.dx, obj_2.dy])
        size_1 = np.array([obj_1.w, obj_1.h])
        size_2 = np.array([obj_2.w, obj_2.h])

        cond1 = np.all(np.abs(pos_1 - pos_2) < (size_1 + size_2) / 2 + 1)
        cond2 = np.all(
            np.abs(pos_1 + speed_1 - pos_2 - speed_2) < (size_1 + size_2) / 2
        )
        return cond1 or cond2


class Object:
    """Represents an object in the OCAtari environment.

    Attributes:
        center (tuple): The center coordinates of the object (x, y).
        w (float): Width of the object.
        h (float): Height of the object.
        dx (float): Horizontal speed of the object.
        dy (float): Vertical speed of the object.
    """

    def __init__(self, x_center, y_center, width, height, dx, dy):
        self.center = (x_center, y_center)
        self.w = width
        self.h = height
        self.dx = dx
        self.dy = dy
