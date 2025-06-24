import numpy as np
import cv2


class ObjectDiscoveryEncoder:
    """Rule-based encoder for Atari games with batched environment support."""

    def __init__(
        self,
        speed_scale: float = 10.0,
        max_objects: int = 256,
        num_envs: int = 1,
    ):
        self.num_envs = num_envs
        self.speed_scale = speed_scale  # Scale to normalize the ball speed
        self.max_objects = max_objects
        self.method = "object_discovery"  # For consistency with other encoders

    def __call__(self, states: np.ndarray) -> np.ndarray:
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
            # colors = []
            for contour in contours:
                areas.append(cv2.contourArea(contour))
                x, y, w, h = cv2.boundingRect(contour)
                coords.append([x, y])  # Center of bounding box
                # colors.append(frame[y + h // 2, x + w // 2])  # Color at center of bbox
                aspect_ratios.append(w / h if h > 0 else 0.0)
                lengths.append(cv2.arcLength(contour, closed=True))
                is_convex.append(float(cv2.isContourConvex(contour)))

            coords_np = np.array(coords, dtype=np.float32).reshape(-1, 1, 2)
            areas_np = np.array(areas, dtype=np.float32)
            aspect_ratios_np = np.array(aspect_ratios, dtype=np.float32)
            lengths_np = np.array(lengths, dtype=np.float32)
            is_convex_np = np.array(is_convex, dtype=np.float32)
            # colors_np = np.array(colors, dtype=np.float32)

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
            feature_vectors = np.zeros((len(contours), 8), dtype=np.float32)
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
            # feature_vectors[:, 8] = (
            #    2 * colors_np / 255.0 - 1
            # )  # Color normalized to [-1, 1]

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
                8,
            ), f"Feature vector shape mismatch: {feature_vectors.shape}"

            batch_features.append(feature_vectors)

        return np.stack(batch_features)  # [num_envs, feature_dim, vector_dim]
