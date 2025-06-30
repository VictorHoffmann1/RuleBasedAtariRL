import numpy as np
import cv2
from typing import List, Optional


class BreakoutEncoder:
    """Rule-based encoder for Atari games with batched environment support."""

    def __init__(
        self,
        encoding_method: str = "paddle+ball",
        speed_scale: float = 10.0,
        num_envs: int = 1,
    ):
        frame_x_size = 160
        num_brick_layers = 6
        num_bricks_per_layer = 18
        self.method = encoding_method
        self.bricks_y_start, self.bricks_y_end = 26, 62
        bricks_shape = (self.bricks_y_end - self.bricks_y_start, frame_x_size)
        self.num_brick_layers = 6
        self.num_bricks_per_layer = 18
        self.brick_x_length = bricks_shape[1] // num_bricks_per_layer
        self.brick_y_length = bricks_shape[0] // num_brick_layers
        self.num_envs = num_envs
        self.ball_x, self.ball_y = np.zeros(num_envs), np.zeros(num_envs)
        self.ball_dx, self.ball_dy = np.zeros(num_envs), np.zeros(num_envs)
        self.speed_scale = speed_scale  # Scale to normalize the ball speed
        self.area_scale = 16
        self.last_paddle_x = np.zeros(self.num_envs)  # Track last paddle position
        self.player_y = 2 * 158 / 179 - 1
        self.players_dx = np.zeros(self.num_envs)  # Track paddle speed
        self.max_objects = 110
        self.reset()

    def reset(self, indices: Optional[List[int]] = None):
        """Resets the encoder state for specific environments.
        Args:
            indices: List of environment indices to reset. If None, resets all.
        """
        if indices is None:
            indices = list(range(self.num_envs))

        for i in indices:
            self.ball_x[i] = self.ball_y[i] = -2.0
            self.ball_dx[i] = self.ball_dy[i] = 0.0

    def __call__(self, states: np.ndarray) -> np.ndarray:
        """Encodes a batch of frames into feature spaces by return paddle position and ball position + velocity.
        Args:
            states: Batch of input frames [num_envs, H, W, C]
        Returns:
            np.ndarray: Batch of encoded features [num_envs, feature_dim]
        """
        batch_features = []

        for i, state in enumerate(states):
            # Crop the frame
            frame = state[31:, 8:-8]

            edges = cv2.adaptiveThreshold(
                frame,
                maxValue=255,
                adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                thresholdType=cv2.THRESH_BINARY,
                blockSize=3,
                C=0,
            )

            # Find contours in the binary mask
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            ball_found = False
            player_x = self.last_paddle_x[i]  # Default to last known paddle position

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                x_norm = 2 * x / frame.shape[1] - 1
                y_norm = 2 * y / frame.shape[0] - 1

                # Detect the player
                if h == 4 and y == 158:
                    player_x = x_norm
                    self.players_dx[i] = x_norm - self.last_paddle_x[i]
                    self.last_paddle_x[i] = player_x

                # Detect the ball
                elif w == 2 and h == 4:
                    ball_found = True
                    if self.ball_x[i] != -2.0 or self.ball_y[i] != -2.0:
                        self.ball_dx[i] = x_norm - self.ball_x[i]
                        self.ball_dy[i] = y_norm - self.ball_y[i]
                    self.ball_x[i] = x_norm
                    self.ball_y[i] = y_norm

            # Handle missing ball
            if not ball_found:
                # Make sure the ball is not out of bounds
                if abs(self.ball_y[i]) >= 1.0:
                    self.reset([i])
                else:
                    self.ball_x[i] += self.ball_dx[i]
                    self.ball_y[i] += self.ball_dy[i]
                    self.ball_x[i] = np.clip(self.ball_x[i], -1.0, 1.0)
                    self.ball_y[i] = np.clip(self.ball_y[i], -1.1, 1.1)

            if self.method == "paddle+ball":
                # Build feature vector
                features = np.array(
                    [
                        player_x,
                        self.ball_x[i],
                        self.ball_y[i],
                        self.ball_dx[i] * self.speed_scale,
                        self.ball_dy[i] * self.speed_scale,
                    ]
                )

            elif self.method == "paddle-ball": # Egocentric method
                features = np.array(
                    [
                        self.ball_x[i] - player_x,
                        self.ball_y[i] - self.player_y,
                        self.ball_dx[i] * self.speed_scale,
                        self.ball_dy[i] * self.speed_scale,
                    ]
                )

            # Brick detection
            elif "bricks" in self.method:
                # Extract the bricks zone from the frame
                bricks_zone = frame[self.bricks_y_start : self.bricks_y_end, :]
                # Create a mask for bricks (assuming bricks are colored differently)
                brick_mask = bricks_zone[:, :] > 0
                # Reshape the brick_mask into a grid of layers and bricks
                reshaped_mask = brick_mask.reshape(
                    self.num_brick_layers,
                    self.brick_y_length,
                    self.num_bricks_per_layer,
                    self.brick_x_length,
                )
                # Use NumPy's any() along the appropriate axes to determine if any pixel in each brick is True
                bricks = reshaped_mask.all(axis=(1, 3))

                if "object_vectors" in self.method:
                    # For deep sets, we will encode each object as a feature vector (x,y, dx, dy, isactive, is_player, is_ball, is_brick)
                    player_vector = np.array(
                        [player_x, self.player_y, self.players_dx[i], 0, 1, 1, 0, 0]
                    )
                    ball_vector = np.array(
                        [
                            self.ball_x[i],
                            self.ball_y[i],
                            self.ball_dx[i] * self.speed_scale,
                            self.ball_dy[i] * self.speed_scale,
                            1 if self.ball_x[i] != -2.0 else 0,
                            0,  # is_player
                            1,  # is_ball
                            0,  # is_brick
                        ]
                    )
                    # Create coordinate grids for vectorized computation
                    j_indices, k_indices = np.meshgrid(
                        np.arange(self.num_brick_layers),
                        np.arange(self.num_bricks_per_layer),
                        indexing="ij",
                    )

                    # Vectorized computation of brick positions and states
                    bricks_vectors = np.zeros(
                        (self.num_brick_layers, self.num_bricks_per_layer, 8)
                    )
                    bricks_vectors[:, :, 0] = (
                        k_indices * self.brick_x_length / frame.shape[1] * 2 - 1
                    )  # x positions
                    bricks_vectors[:, :, 1] = (
                        j_indices * self.brick_y_length / frame.shape[0] * 2 - 1
                    )  # y positions
                    bricks_vectors[:, :, 4] = bricks.astype(np.float32)  # is active
                    bricks_vectors[:, :, 7] = 1  # is_brick

                    # Concatenate player, ball, and bricks vectors
                    features = np.concatenate(
                        [
                            player_vector.reshape(1, -1),
                            ball_vector.reshape(1, -1),
                            bricks_vectors.reshape(-1, len(player_vector)),
                        ],
                        axis=0,
                    )

                elif "discovery" in self.method:
                    # For deep sets, we will encode each object as a feature vector (x,y, dx, dy, w, h) for active objects
                    player_vector = np.array(
                        [
                            player_x,
                            self.player_y,
                            self.players_dx[i],
                            0,
                            16 / self.area_scale,
                            4 / self.area_scale,
                        ]
                    )
                    ball_vector = (
                        np.array(
                            [
                                self.ball_x[i],
                                self.ball_y[i],
                                self.ball_dx[i] * self.speed_scale,
                                self.ball_dy[i] * self.speed_scale,
                                2 / self.area_scale,
                                4 / self.area_scale,
                            ]
                        )
                        if self.ball_x[i] != -2.0
                        else np.zeros(6, dtype=np.float32)
                    )
                    # Create coordinate grids for vectorized computation
                    j_indices, k_indices = np.meshgrid(
                        np.arange(self.num_brick_layers),
                        np.arange(self.num_bricks_per_layer),
                        indexing="ij",
                    )

                    # Vectorized computation of brick positions and states
                    bricks_vectors = np.zeros(
                        (self.num_brick_layers, self.num_bricks_per_layer, 6)
                    )
                    # Only include bricks that are active
                    bricks_vectors[bricks, 0] = (
                        k_indices[bricks] * self.brick_x_length / frame.shape[1] * 2 - 1
                    )  # x positions
                    bricks_vectors[bricks, 1] = (
                        j_indices[bricks] * self.brick_y_length / frame.shape[0] * 2 - 1
                    )  # y positions
                    bricks_vectors[bricks, 4] = self.brick_x_length / self.area_scale
                    bricks_vectors[bricks, 5] = self.brick_y_length / self.area_scale

                    # Concatenate player, ball, and bricks vectors
                    features = np.concatenate(
                        [
                            player_vector.reshape(1, -1),
                            ball_vector.reshape(1, -1),
                            bricks_vectors.reshape(-1, len(player_vector)),
                        ],
                        axis=0,
                    )

                else:
                    # Build feature vector
                    features = np.concatenate(
                        [
                            np.array(
                                [
                                    player_x,
                                    self.ball_x[i],
                                    self.ball_y[i],
                                    self.ball_dx[i] * self.speed_scale,
                                    self.ball_dy[i] * self.speed_scale,
                                ]
                            ),
                            bricks.flatten(),
                        ]
                    )

            else:
                raise ValueError(f"Unknown method: {self.method}")

            batch_features.append(features)
        return np.stack(batch_features)  # [num_envs, feature_dim]
