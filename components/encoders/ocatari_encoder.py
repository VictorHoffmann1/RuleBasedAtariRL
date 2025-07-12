import numpy as np
import gymnasium as gym
from ocatari.core import OCAtari
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

        # Single normalizer for all dynamic features: dx, dy, width, height
        self.feature_normalizer = RunningNormalizer(num_vars=4)  # dx, dy, width, height

        # Add caching for object extraction to avoid repeated wrapper traversal
        self._env_ocatari_cache = {}

        # Performance optimization flags
        self.lazy_normalize = True  # Only normalize when statistics have stabilized
        self.min_samples_for_norm = 100  # Minimum samples before normalizing

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
        num_envs = len(envs.envs)

        if self.method == "discovery":
            # Pre-allocate the entire batch array
            batch_features = np.zeros((num_envs, self.max_objects, self.n_features))

            # Cache OCAtari objects for all environments
            all_objects = [self.get_ocatari_objects(env) for env in envs.envs]

            # Process environments in vectorized manner where possible
            for env_idx, objects in enumerate(all_objects):
                valid_objects = [obj for obj in objects if obj.category != "NoObject"]
                num_valid = min(len(valid_objects), self.max_objects)

                if num_valid == 0:
                    continue

                if len(valid_objects) > self.max_objects:
                    warnings.warn(
                        f"More objects than max_objects ({self.max_objects}) in environment. Truncating.",
                        UserWarning,
                    )
                    valid_objects = valid_objects[: self.max_objects]

                # Extract all object data in vectorized operations
                centers_x = np.array([obj.center[0] for obj in valid_objects])
                centers_y = np.array([obj.center[1] for obj in valid_objects])
                dynamic_data = np.array(
                    [[obj.dx, obj.dy, obj.w, obj.h] for obj in valid_objects]
                )

                # Vectorized normalization for positions
                norm_x = self._normalize(centers_x, self.img_width, True)
                norm_y = self._normalize(centers_y, self.img_height, True)

                # Batch normalize dynamic features
                if num_valid > 0:
                    normalized_dynamic = self._batch_normalize_dynamic(dynamic_data)

                    # Build feature matrix efficiently
                    object_features = np.column_stack(
                        [
                            norm_x,
                            norm_y,
                            normalized_dynamic[:, 0],  # dx
                            normalized_dynamic[:, 1],  # dy
                            normalized_dynamic[:, 2],  # width
                            normalized_dynamic[:, 3],  # height
                        ]
                    )

                    # Handle RGB features if needed
                    if self.use_rgb:
                        rgb_data = np.array([obj.rgb for obj in valid_objects]) / 255.0
                        rgb_features = 2 * rgb_data - 1  # Normalize to [-1, 1]
                        object_features = np.column_stack(
                            [object_features, rgb_features]
                        )

                    # Handle category features if needed
                    if self.use_category:
                        category_features = self._get_category_features_batch(
                            valid_objects
                        )
                        object_features = np.column_stack(
                            [object_features, category_features]
                        )

                    batch_features[env_idx, :num_valid] = object_features

        elif self.method == "expert":
            batch_features = np.full((num_envs, self.n_features), -2.0)
            batch_features[:, 0] = 0.0  # Initialize paddle position to 0

            for env_idx, env in enumerate(envs.envs):
                objects = self.get_ocatari_objects(env)
                player_found, ball_found = False, False

                for obj in objects:
                    if obj.category == "Player":
                        batch_features[env_idx, 0] = self._normalize(
                            obj.x, self.img_width, True
                        )
                        player_found = True
                    elif obj.category == "Ball":
                        batch_features[env_idx, 1] = self._normalize(
                            obj.x, self.img_width, True
                        )
                        batch_features[env_idx, 2] = self._normalize(
                            obj.y, self.img_height, True
                        )
                        # Batch normalize velocity
                        normalized_velocity = self.feature_normalizer(
                            [obj.dx, obj.dy, 0, 0]
                        )
                        batch_features[env_idx, 3] = normalized_velocity[0]
                        batch_features[env_idx, 4] = normalized_velocity[1]
                        ball_found = True

                    if player_found and ball_found:
                        break

        return batch_features

    def _get_category_features_batch(self, objects):
        """Generate category features for multiple objects efficiently."""
        num_objects = len(objects)
        category_features = np.zeros((num_objects, 3))

        for i, obj in enumerate(objects):
            if obj.category == "Player":
                category_features[i, 0] = 1.0
            elif obj.category == "Ball":
                category_features[i, 1] = 1.0
            else:
                category_features[i, 2] = 1.0

        return category_features

    # Cache for OCAtari object lookup to avoid repeated wrapper traversal
    _ocatari_cache = {}

    @classmethod
    def get_ocatari_objects(cls, env: gym.Env):
        """
        Get the objects from the underlying OCAtari environment through the wrapper chain.
        Uses caching to avoid repeated wrapper traversal.

        :param env: The wrapped environment
        :return: The objects from the OCAtari environment
        """
        env_id = id(env)

        # Check cache first
        if env_id in cls._ocatari_cache:
            return cls._ocatari_cache[env_id].objects

        # Traverse through wrappers to find the OCAtari instance
        current_env = env
        while hasattr(current_env, "env"):
            if isinstance(current_env, OCAtari):
                cls._ocatari_cache[env_id] = current_env
                return current_env.objects
            current_env = current_env.env

        # Check if the current env is OCAtari
        if isinstance(current_env, OCAtari):
            cls._ocatari_cache[env_id] = current_env
            return current_env.objects

        # If we can't find OCAtari, raise an error
        raise ValueError("No OCAtari environment found in the wrapper chain")

    @staticmethod
    def _normalize(values, scale, centering=False):
        """Vectorized version of normalize for batch processing."""
        if centering:
            return np.clip(2 * (values / scale) - 1, -1, 1)
        else:
            return np.clip(values / scale, -1, 1)

    def _batch_normalize_dynamic(self, dynamic_data):
        """Efficiently normalize dynamic features for multiple objects."""
        if len(dynamic_data) == 0:
            return dynamic_data

        # For better performance, update statistics in batch
        for row in dynamic_data:
            self.feature_normalizer.sum += row
            self.feature_normalizer.count += 1
            self.feature_normalizer.sum_squared += row**2

        # Mark cache as invalid
        self.feature_normalizer._cache_valid = False

        # Use lazy normalization - only normalize if we have enough samples
        if (
            self.lazy_normalize
            and np.min(self.feature_normalizer.count) < self.min_samples_for_norm
        ):
            # Just apply basic clipping without normalization during warmup
            return np.clip(dynamic_data / 10.0, -1, 1)  # Simple scaling

        # Get current stats (uses caching)
        mean_vals = self.feature_normalizer.mean()
        std_vals = self.feature_normalizer.std()

        # Vectorized normalization
        normalized = (dynamic_data - mean_vals) / (
            std_vals + self.feature_normalizer.epsilon
        )
        return np.clip(normalized, -10, 10)

    def warm_up_normalizer(self, sample_envs, num_warmup_steps=100):
        """
        Warm up the normalizer with sample data for better initial performance.
        Call this once before training to avoid cold start normalization issues.
        """
        print(f"Warming up normalizer with {num_warmup_steps} steps...")
        for _ in range(num_warmup_steps):
            try:
                # Generate sample features to warm up the normalizer
                _ = self(sample_envs)  # Discarded result, just for warmup
            except Exception as e:
                print(f"Warmup step failed: {e}")
                break
        print("Normalizer warmup complete.")

    def get_cache_stats(self):
        """Return statistics about the internal caches for debugging."""
        return {
            "ocatari_cache_size": len(self._ocatari_cache),
            "normalizer_samples": int(np.min(self.feature_normalizer.count))
            if hasattr(self.feature_normalizer, "count")
            else 0,
            "cache_warmed_up": np.min(self.feature_normalizer.count)
            >= self.min_samples_for_norm
            if hasattr(self.feature_normalizer, "count")
            else False,
        }


class RunningNormalizer:
    """Optimized running normalizer that can handle multiple variables simultaneously."""

    def __init__(self, num_vars: int = 1, epsilon: float = 1e-8):
        self.num_vars = num_vars
        self.sum = np.zeros(num_vars, dtype=np.float64)
        self.count = np.zeros(num_vars, dtype=np.int64)
        self.sum_squared = np.zeros(num_vars, dtype=np.float64)
        self.epsilon = epsilon

        # Cache frequently used values
        self._mean_cache = None
        self._std_cache = None
        self._cache_valid = False

    def __call__(self, x):
        """Normalize single value or array of values for multiple variables."""
        x = np.asarray(x, dtype=np.float32)

        if x.ndim == 0:  # Single scalar for all variables
            x = np.full(self.num_vars, x, dtype=np.float32)
        elif x.size != self.num_vars:
            raise ValueError(f"Expected {self.num_vars} values, got {x.size}")

        x = x.reshape(self.num_vars).astype(np.float32)

        # Update running statistics for each variable
        self.sum += x
        self.count += 1
        self.sum_squared += x**2
        self._cache_valid = False

        # Vectorized normalization
        valid_mask = self.count > 0
        mean_vals = np.where(valid_mask, self.sum / self.count, 0.0)

        std_mask = self.count > 1
        std_vals = np.zeros(self.num_vars)
        std_vals[std_mask] = np.sqrt(
            (
                self.sum_squared[std_mask]
                - (self.sum[std_mask] ** 2) / self.count[std_mask]
            )
            / (self.count[std_mask] - 1)
        )

        normalized = (x - mean_vals) / (std_vals + self.epsilon)
        return np.clip(normalized, -10, 10)

    def mean(self):
        """Return the mean of each variable with caching."""
        if not self._cache_valid:
            self._update_cache()
        return self._mean_cache

    def std(self):
        """Return the standard deviation of each variable with caching."""
        if not self._cache_valid:
            self._update_cache()
        return self._std_cache

    def _update_cache(self):
        """Update cached mean and std values."""

        self._mean_cache = np.where(self.count > 0, self.sum / self.count, 0.0)
        valid_mask = self.count > 1
        self._std_cache = np.zeros(self.num_vars)
        self._std_cache[valid_mask] = np.sqrt(
            (
                self.sum_squared[valid_mask]
                - (self.sum[valid_mask] ** 2) / self.count[valid_mask]
            )
            / (self.count[valid_mask] - 1)
        )
        self._cache_valid = True
