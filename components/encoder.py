import numpy as np
import cv2
from typing import List, Optional
import numpy as np


class RuleBasedEncoder:
    """Rule-based encoder for Atari games with batched environment support."""

    def __init__(
        self,
        encoding_method: str = "paddle+ball",
        num_brick_layers: int = 6,
        num_bricks_per_layer: int = 18,
        bricks_y_start: int = 26,
        bricks_y_end: int = 62,
        frame_x_size: int = 160,
        speed_scale: float = 10.0,
        max_objects: int = 256,
        num_envs: int = 1,
    ):
        self.method = encoding_method
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
        self.max_objects = max_objects
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

    def simple_method(
        self, states: np.ndarray, method: str = "paddle+ball"
    ) -> np.ndarray:
        """Encodes a batch of frames into feature spaces by return paddle position and ball position + velocity.
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

            if method == "paddle+ball":
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

            # Brick detection
            elif method == "bricks+paddle+ball":
                # Extract the bricks zone from the frame
                bricks_zone = frame[self.bricks_y_start : self.bricks_y_end, :, :]
                # Create a mask for bricks (assuming bricks are colored differently)
                brick_mask = (
                    (bricks_zone[:, :, 0] > 0)
                    & (bricks_zone[:, :, 1] > 0)
                    & (bricks_zone[:, :, 2] > 0)
                )
                # Reshape the brick_mask into a grid of layers and bricks
                reshaped_mask = brick_mask.reshape(
                    self.num_brick_layers,
                    self.brick_y_length,
                    self.num_bricks_per_layer,
                    self.brick_x_length,
                )
                # Use NumPy's any() along the appropriate axes to determine if any pixel in each brick is True
                bricks = reshaped_mask.all(axis=(1, 3))

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
                raise ValueError(f"Unknown method: {method}")

            batch_features.append(features)

        return np.stack(batch_features)  # [num_envs, feature_dim]

    def build_feature_vectors(self, states: np.ndarray) -> np.ndarray:
        """Encodes a batch of frames into feature vectors that can be fed to transformers.
        For each detected feature, a vector containing the position, speed and area of the object is returned.
        Args:
            states: Batch of input frames [num_envs, H, W, C]
        Returns:
            np.ndarray: Batch of encoded features [num_envs, feature_dim, vector_dim]
        """
        # TODO: Input a stack of 2 consecutive frames for each environment
        batch_features = []

        for i, state in enumerate(states):
            # Crop the frame
            if state.ndim == 3:
                frame = state[31:, 8:-8, 0]
                next_frame = state[31:, 8:-8, 1]  # used to compute speed
            else:
                raise ValueError(
                    f"Expected input state to be of shape [num_envs, H, W, C], but got {state.shape}"
                )

            # Use adaptive thresholding to get an edge / binary mask
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

            # Compute coords, areas, aspect ratios, lengths and convexity for contours
            coords = []
            areas = []
            aspect_ratios = []
            lengths = []
            is_convex = []
            colors = []
            for contour in contours:
                areas.append(cv2.contourArea(contour))
                x, y, w, h = cv2.boundingRect(contour)
                coords.append([x, y])  # Center of bounding box
                colors.append(frame[y + h // 2, x + w // 2])  # Color at center of bbox
                aspect_ratios.append(w / h if h > 0 else 0.0)
                lengths.append(cv2.arcLength(contour, closed=True))
                is_convex.append(float(cv2.isContourConvex(contour)))

            coords_np = np.array(coords, dtype=np.float32).reshape(-1, 1, 2)
            areas_np = np.array(areas, dtype=np.float32)
            aspect_ratios_np = np.array(aspect_ratios, dtype=np.float32)
            lengths_np = np.array(lengths, dtype=np.float32)
            is_convex_np = np.array(is_convex, dtype=np.float32)
            colors_np = np.array(colors, dtype=np.float32)

            assert len(coords_np) == len(contours), (
                f"Number of coords {len(coords_np)} does not match number of contours {len(contours)}"
            )

            # Calculate optical flow (Lucas-Kanade) for coords
            if len(coords_np) > 0:
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    frame, next_frame, coords_np, None, winSize=(7, 7), maxLevel=2
                )
                # Compute speed as difference between new and old positions
                speeds = (
                    (next_pts - coords_np).reshape(-1, 2)
                    if next_pts is not None
                    else np.zeros_like(coords_np.reshape(-1, 2))
                )
            else:
                speeds = np.zeros((0, 2), dtype=np.float32)

            assert len(speeds) == len(contours), (
                f"Number of speeds {len(speeds)} does not match number of contours {len(contours)}"
            )

            # Extract features for each contour (centroid, speed, area, aspect ratio, length, isConvex, color)
            feature_vectors = np.zeros((len(contours), 9), dtype=np.float32)
            feature_vectors[:, 0:2] = (
                2
                * coords_np.reshape(-1, 2)
                / np.array([frame.shape[1], frame.shape[0]])
                - 1
            )
            feature_vectors[:, 2:4] = (
                speeds * self.speed_scale / np.array([frame.shape[1], frame.shape[0]])
            )  # Normalize speed

            feature_vectors[:, 4] = np.log1p(areas_np)  # Area (log-scaled)
            feature_vectors[:, 5] = np.log(aspect_ratios_np + 1e-8)  # Aspect ratio
            feature_vectors[:, 6] = np.log1p(lengths_np)  # Length (log-scaled)
            feature_vectors[:, 7] = 2 * is_convex_np - 1  # Is convex (-1 or 1)
            feature_vectors[:, 8] = (
                2 * colors_np / 255.0 - 1
            )  # Color normalized to [-1, 1]

            # Pad / Truncate to max_objects
            if len(feature_vectors) < self.max_objects:
                padding = np.zeros(
                    (self.max_objects - len(feature_vectors), feature_vectors.shape[1]),
                    dtype=np.float32,
                )
                feature_vectors = np.vstack((feature_vectors, padding))
            elif len(feature_vectors) > self.max_objects:
                feature_vectors = feature_vectors[: self.max_objects]

            # Ensure the feature vector has the correct shape
            assert feature_vectors.shape == (
                self.max_objects,
                9,
            ), f"Feature vector shape mismatch: {feature_vectors.shape}"

            batch_features.append(feature_vectors)

        return np.stack(batch_features)  # [num_envs, feature_dim, vector_dim]

    def __call__(self, states: np.ndarray) -> np.ndarray:
        """Encodes a batch of frames into feature spaces.
        Args:
            states: Batch of input frames [num_envs, H, W, C]
        Returns:
            np.ndarray: Batch of encoded features [num_envs, feature_dim]
        """

        if self.method in ["paddle+ball", "bricks+paddle+ball"]:
            return self.simple_method(states, method=self.method)
        elif self.method == "transformer":
            return self.build_feature_vectors(states)
        else:
            raise ValueError(f"Unknown encoding method: {self.method}")
