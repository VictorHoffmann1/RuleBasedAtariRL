import numpy as np
import cv2
from typing import List, Optional
import numpy as np


class RuleBasedEncoder:
    """Rule-based encoder for Atari games with batched environment support."""

    def __init__(
        self,
        num_brick_layers: int = 6,
        num_bricks_per_layer: int = 18,
        bricks_y_start: int = 26,
        bricks_y_end: int = 62,
        frame_x_size: int = 160,
        speed_scale: float = 10.0,
        num_envs: int = 1,
    ):
        self.bricks_y_start, self.bricks_y_end = int(bricks_y_start), int(bricks_y_end)
        bricks_shape = (self.bricks_y_end - self.bricks_y_start, frame_x_size)
        self.num_brick_layers = num_brick_layers
        self.num_bricks_per_layer = num_bricks_per_layer
        self.brick_x_length = bricks_shape[1] // num_bricks_per_layer
        self.brick_y_length = bricks_shape[0] // num_brick_layers
        self.num_envs = num_envs
        self.ball_x, self.ball_y = np.zeros(num_envs), np.zeros(num_envs)
        self.ball_dx, self.ball_dy = np.zeros(num_envs), np.zeros(num_envs)
        self.speed_scale = speed_scale  # Scale to normalize the ball speed
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
        """Encodes a batch of frames into feature spaces.
        Args:
            states: Batch of input frames [num_envs, H, W, C]
        Returns:
            np.ndarray: Batch of encoded features [num_envs, feature_dim]
        """
        batch_features = []

        for i, state in enumerate(states):
            # Crop the frame
            frame = state[31:, 8:-8] if state.ndim == 3 else state[0, 31:, 8:-8]

            # Get the ball and player information (they both have the same color)
            color_mask = (frame[:, :, 0] > 195) & (frame[:, :, 1] < 80)
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
                color_mask.astype(np.uint8)
            )

            ball_found = False
            player_x = 0.0  # Default center position if not found

            for j in range(1, num_labels):  # skip label 0 (background)
                x, y, _, _, area = stats[j]
                x_norm = 2 * x / frame.shape[1] - 1
                y_norm = 2 * y / frame.shape[0] - 1

                # Detect the player
                if 31 < area < 80:
                    player_x = x_norm

                # Detect the ball
                elif 2 < area < 15:
                    ball_found = True
                    if self.ball_x[i] != -2.0 or self.ball_y[i] != -2.0:
                        self.ball_dx[i] = x_norm - self.ball_x[i]
                        self.ball_dy[i] = y_norm - self.ball_y[i]
                    self.ball_x[i] = x_norm
                    self.ball_y[i] = y_norm

            # Handle Missing Ball
            if not ball_found:
                # Make sure the ball is not out of bounds
                if self.ball_y[i] >= 1.0:
                    self.reset([i])
                self.ball_x[i] += self.ball_dx[i]
                self.ball_y[i] += self.ball_dy[i]

            # Brick detection
            # bricks_zone = frame[self.bricks_y_start:self.bricks_y_end, :, :]
            # brick_mask = (bricks_zone[:, :, 0] > 0) & (bricks_zone[:, :, 1] > 0) & (bricks_zone[:, :, 2] > 0)
            # Reshape the brick_mask into a grid of layers and bricks
            # reshaped_mask = brick_mask.reshape(self.num_brick_layers, self.brick_y_length, self.num_bricks_per_layer, self.brick_x_length)
            # Use NumPy's any() along the appropriate axes to determine if any pixel in each brick is True
            # bricks = reshaped_mask.all(axis=(1, 3))

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
                    # bricks.flatten()
                ]
            )
            batch_features.append(features)

        return np.stack(batch_features)  # [num_envs, feature_dim]
