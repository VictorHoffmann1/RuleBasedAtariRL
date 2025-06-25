import numpy as np
import cv2
from typing import List, Optional


class PongEncoder:
    """Rule-based encoder for Atari games with batched environment support."""

    def __init__(
        self,
        encoding_method: str = "paddle+ball",  # Not used
        speed_scale: float = 10.0,
        num_envs: int = 1,
    ):
        self.method = encoding_method
        self.num_envs = num_envs
        self.ball_x, self.ball_y = np.zeros(num_envs), np.zeros(num_envs)
        self.ball_dx, self.ball_dy = np.zeros(num_envs), np.zeros(num_envs)
        self.speed_scale = speed_scale  # Scale to normalize the ball speed
        self.last_player_y = np.zeros(self.num_envs)  # Track last paddle position
        self.last_oppopent_y = np.zeros(
            self.num_envs
        )  # Track last opponent paddle position
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
            player_y = self.last_player_y[i]  # Default to last known paddle position
            opponent_y = self.last_oppopent_y[
                i
            ]  # Default to last known opponent paddle position

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                x_norm = 2 * x / frame.shape[1] - 1
                y_norm = 2 * y / frame.shape[0] - 1

                # Detect the player
                if w == 4 and x == 140:
                    player_y = y_norm
                    self.last_player_y[i] = player_y

                # Detect the ball
                elif w == 2 and h == 4:
                    ball_found = True
                    if self.ball_x[i] != -2.0 or self.ball_y[i] != -2.0:
                        self.ball_dx[i] = x_norm - self.ball_x[i]
                        self.ball_dy[i] = y_norm - self.ball_y[i]
                    self.ball_x[i] = x_norm
                    self.ball_y[i] = y_norm

                # Detect the opponent paddle
                elif w == 4 and x == 16:
                    self.last_oppopent_y[i] = y_norm
                    opponent_y = y_norm
                    # Note: The opponent paddle position is not used in the feature vector

            # Handle missing ball
            if not ball_found:
                # Make sure the ball is not out of bounds
                if self.ball_y[i] >= 1.0:
                    self.reset([i])
                else:
                    self.ball_x[i] += self.ball_dx[i]
                    self.ball_y[i] += self.ball_dy[i]
                    self.ball_x[i] = np.clip(self.ball_x[i], -1.0, 1.0)
                    self.ball_y[i] = np.clip(self.ball_y[i], -1.0, 1.1)

            # Build feature vector
            features = np.array(
                [
                    player_y,
                    opponent_y,
                    self.ball_x[i],
                    self.ball_y[i],
                    self.ball_dx[i] * self.speed_scale,
                    self.ball_dy[i] * self.speed_scale,
                ]
            )
            batch_features.append(features)

        return np.stack(batch_features)  # [num_envs, feature_dim]
