import numpy as np
import cv2
from typing import List, Optional


class BreakoutEncoder:
    """
    Rule-based encoder for Atari Breakout game with batched environment support.
    
    This encoder extracts game state features from raw frames including:
    - Paddle position and velocity
    - Ball position and velocity
    - Brick states (for relevant encoding methods)
    
    Supports multiple encoding methods:
    - "paddle+ball": Basic paddle and ball features
    - "paddle-ball": Egocentric (relative) features
    - "paddle+ball+trajectory": Includes ball trajectory prediction
    - "*bricks*": Methods that include brick state information
    """

    # Game constants
    FRAME_WIDTH = 160
    FRAME_HEIGHT = 210
    CROP_TOP = 31
    CROP_SIDE = 8
    
    PADDLE_HEIGHT = 4
    PADDLE_Y = 158
    PADDLE_WIDTH = 16
    
    BALL_WIDTH = 2
    BALL_HEIGHT = 4
    
    BRICKS_Y_START = 26
    BRICKS_Y_END = 62
    NUM_BRICK_LAYERS = 6
    NUM_BRICKS_PER_LAYER = 18
    
    def __init__(
        self,
        encoding_method: str = "paddle+ball",
        speed_scale: float = 10.0,
        num_envs: int = 1,
    ):
        """
        Initialize the BreakoutEncoder.
        
        Args:
            encoding_method: The method to use for encoding game state
            speed_scale: Scale factor for normalizing velocities
            num_envs: Number of parallel environments
        """
        self.method = encoding_method
        self.num_envs = num_envs
        self.speed_scale = speed_scale
        self.area_scale = 16
        self.max_objects = 110
        
        # Calculate brick dimensions
        bricks_height = self.BRICKS_Y_END - self.BRICKS_Y_START
        self.brick_x_length = (self.FRAME_WIDTH - 2 * self.CROP_SIDE) // self.NUM_BRICKS_PER_LAYER
        self.brick_y_length = bricks_height // self.NUM_BRICK_LAYERS
        
        # Initialize tracking arrays for all environments
        self._init_tracking_arrays()
        
        # Normalized paddle Y position (constant)
        self.player_y_norm = self.normalize(self.PADDLE_Y, self.FRAME_HEIGHT - self.CROP_TOP, centering=True)

        self.reset()
    
    def _init_tracking_arrays(self):
        """Initialize arrays to track game state across environments."""
        self.ball_x = np.zeros(self.num_envs)
        self.ball_y = np.zeros(self.num_envs)
        self.ball_dx = np.zeros(self.num_envs)
        self.ball_dy = np.zeros(self.num_envs)
        self.last_paddle_x = np.zeros(self.num_envs)
        self.paddle_dx = np.zeros(self.num_envs)

    def reset(self, indices: Optional[List[int]] = None):
        """
        Reset the encoder state for specific environments.
        
        Args:
            indices: List of environment indices to reset. If None, resets all.
        """
        if indices is None:
            indices = list(range(self.num_envs))

        for i in indices:
            self.ball_x[i] = 0.0
            self.ball_y[i] = 0.0
            self.ball_dx[i] = 0.0
            self.ball_dy[i] = 0.0

    def __call__(self, states: np.ndarray, eps=0.1) -> np.ndarray:
        """
        Encode a batch of frames into feature vectors.
        
        Args:
            states: Batch of input frames [num_envs, H, W, C]
            eps: Unused parameter (kept for compatibility)
            
        Returns:
            np.ndarray: Batch of encoded features [num_envs, feature_dim]
        """
        batch_features = []

        for i, state in enumerate(states):
            # Extract game objects from the current frame
            frame = self._preprocess_frame(state)
            paddle_x, ball_pos, ball_vel = self._extract_objects(frame, i)
            
            # Generate features based on the selected method
            features = self._generate_features(frame, paddle_x, ball_pos, ball_vel, i)
            batch_features.append(features)
            
        return np.stack(batch_features)
    
    def _preprocess_frame(self, state: np.ndarray) -> np.ndarray:
        """Crop and preprocess the input frame."""
        return state[self.CROP_TOP:, self.CROP_SIDE:-self.CROP_SIDE]
    
    def _extract_objects(self, frame: np.ndarray, env_idx: int) -> tuple:
        """
        Extract game objects (paddle, ball) from the frame.
        
        Args:
            frame: Preprocessed frame
            env_idx: Environment index
            
        Returns:
            tuple: (paddle_x, ball_position, ball_velocity)
        """
        # Apply edge detection
        edges = cv2.adaptiveThreshold(
            frame,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=3,
            C=0,
        )

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Extract paddle and ball information
        paddle_x = self._extract_paddle(contours, env_idx)
        ball_pos, ball_vel = self._extract_ball(contours, frame, env_idx)
        
        return paddle_x, ball_pos, ball_vel
    
    def _extract_paddle(self, contours: List, env_idx: int) -> float:
        """Extract paddle position from contours."""
        paddle_x = self.last_paddle_x[env_idx]  # Default to last known position
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Detect paddle (specific height and y-position)
            if h == self.PADDLE_HEIGHT and y == self.PADDLE_Y:
                self.paddle_dx[env_idx] = x - self.last_paddle_x[env_idx]
                self.last_paddle_x[env_idx] = x
                paddle_x = x
                break
                
        return paddle_x
    
    def _extract_ball(self, contours: List, frame: np.ndarray, env_idx: int) -> tuple:
        """Extract ball position and velocity from contours."""
        ball_found = False
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Detect ball (specific dimensions)
            if w == self.BALL_WIDTH and h == self.BALL_HEIGHT:
                ball_found = True
                
                # Calculate velocity if ball was previously detected
                if not self._is_ball_missing(env_idx):
                    self.ball_dx[env_idx] = x - self.ball_x[env_idx]
                    self.ball_dy[env_idx] = y - self.ball_y[env_idx]
                
                self.ball_x[env_idx] = x
                self.ball_y[env_idx] = y
                break
        
        # Handle missing ball (predict position)
        if not ball_found:
            self._handle_missing_ball(frame, env_idx)
            
        ball_pos = (self.ball_x[env_idx], self.ball_y[env_idx])
        ball_vel = (self.ball_dx[env_idx], self.ball_dy[env_idx])
        
        return ball_pos, ball_vel
    
    def _handle_missing_ball(self, frame: np.ndarray, env_idx: int):
        """Handle case where ball is not detected in current frame."""
        # Reset if ball is out of bounds
        if self.ball_y[env_idx] >= frame.shape[0]:
            self.reset([env_idx])
        else:
            # Predict ball position based on velocity
            self.ball_x[env_idx] += self.ball_dx[env_idx]
            self.ball_y[env_idx] += self.ball_dy[env_idx]

    def _generate_features(self, frame: np.ndarray, paddle_x: float, ball_pos: tuple, ball_vel: tuple, env_idx: int) -> np.ndarray:
        """
        Generate feature vector based on the selected encoding method.
        
        Args:
            frame: Preprocessed frame
            paddle_x: Paddle x position
            ball_pos: Ball position (x, y)
            ball_vel: Ball velocity (dx, dy)
            env_idx: Environment index
            
        Returns:
            np.ndarray: Feature vector for the current environment
        """
        ball_x, ball_y = ball_pos
        ball_dx, ball_dy = ball_vel
        
        # Normalize all values
        paddle_x_norm = self.normalize(paddle_x, frame.shape[1], centering=True)
        paddle_y_norm = self.player_y_norm
        paddle_dx_norm = self.normalize(self.paddle_dx[env_idx], self.speed_scale, centering=False)
        
        ball_x_norm = self.normalize(ball_x, frame.shape[1], centering=True)
        ball_y_norm = self.normalize(ball_y, frame.shape[0], centering=True)
        ball_dx_norm = self.normalize(ball_dx, self.speed_scale, centering=False)
        ball_dy_norm = self.normalize(ball_dy, self.speed_scale, centering=False)
        
        # Generate features based on method
        if self.method == "paddle+ball":
            return self._generate_basic_features(paddle_x_norm, ball_x_norm, ball_y_norm, ball_dx_norm, ball_dy_norm)
        
        elif self.method == "paddle-ball":
            return self._generate_egocentric_features(paddle_x_norm, paddle_y_norm, ball_x_norm, ball_y_norm, ball_dx_norm, ball_dy_norm)
        
        elif self.method == "paddle+ball+trajectory":
            return self._generate_trajectory_features(frame, paddle_x_norm, paddle_y_norm, ball_x_norm, ball_y_norm, ball_dx_norm, ball_dy_norm, env_idx)
        
        elif "bricks" in self.method:
            return self._generate_brick_features(frame, paddle_x_norm, paddle_y_norm, paddle_dx_norm, ball_x_norm, ball_y_norm, ball_dx_norm, ball_dy_norm, env_idx)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _generate_basic_features(self, paddle_x_norm: float, ball_x_norm: float, ball_y_norm: float, ball_dx_norm: float, ball_dy_norm: float) -> np.ndarray:
        """Generate basic paddle+ball features."""
        return np.array([
            paddle_x_norm,
            ball_x_norm,
            ball_y_norm,
            ball_dx_norm,
            ball_dy_norm,
        ])
    
    def _generate_egocentric_features(self, paddle_x_norm: float, paddle_y_norm: float, ball_x_norm: float, ball_y_norm: float, ball_dx_norm: float, ball_dy_norm: float) -> np.ndarray:
        """Generate egocentric (paddle-relative) features."""
        return np.array([
            ball_x_norm - paddle_x_norm,
            ball_y_norm - paddle_y_norm,
            ball_dx_norm,
            ball_dy_norm,
        ])
    
    def _generate_trajectory_features(self, frame: np.ndarray, paddle_x_norm: float, paddle_y_norm: float, ball_x_norm: float, ball_y_norm: float, ball_dx_norm: float, ball_dy_norm: float, env_idx: int) -> np.ndarray:
        """Generate features including ball trajectory prediction."""
        return np.array([
            paddle_x_norm,  # Absolute paddle position
            ball_x_norm,    # Absolute ball x
            ball_y_norm,    # Absolute ball y
            ball_dx_norm,
            ball_dy_norm,
            ball_x_norm - paddle_x_norm,  # Relative ball x
            ball_y_norm - paddle_y_norm,  # Relative ball y
            self.normalize(self.ball_x[env_idx] + self.ball_dx[env_idx], frame.shape[1], True),     # ball position x in 1 step
            self.normalize(self.ball_y[env_idx] + self.ball_dy[env_idx], frame.shape[0], True),     # ball position y in 1 step
            self.normalize(self.ball_x[env_idx] + 2 * self.ball_dx[env_idx], frame.shape[1], True), # ball position x in 2 steps
            self.normalize(self.ball_y[env_idx] + 2 * self.ball_dy[env_idx], frame.shape[0], True), # ball position y in 2 steps
            self.normalize(self.ball_x[env_idx] + 3 * self.ball_dx[env_idx], frame.shape[1], True), # ball position x in 3 steps
            self.normalize(self.ball_y[env_idx] + 3 * self.ball_dy[env_idx], frame.shape[0], True), # ball position y in 3 steps
        ])
    def _generate_brick_features(self, frame: np.ndarray, paddle_x_norm: float, paddle_y_norm: float, paddle_dx_norm: float, ball_x_norm: float, ball_y_norm: float, ball_dx_norm: float, ball_dy_norm: float, env_idx: int) -> np.ndarray:
        """Generate features including brick state information."""
        # Extract brick information
        bricks = self._extract_bricks(frame)
        
        if "object_vectors" in self.method:
            return self._generate_object_vector_features(frame, paddle_x_norm, paddle_y_norm, paddle_dx_norm, ball_x_norm, ball_y_norm, ball_dx_norm, ball_dy_norm, bricks, env_idx)
        
        elif "discovery" in self.method:
            return self._generate_discovery_features(frame, paddle_x_norm, paddle_y_norm, paddle_dx_norm, ball_x_norm, ball_y_norm, ball_dx_norm, ball_dy_norm, bricks, env_idx)
        
        else:
            # Basic brick features
            return np.concatenate([
                np.array([paddle_x_norm, ball_x_norm, ball_y_norm, ball_dx_norm, ball_dy_norm]),
                bricks.flatten(),
            ])
    
    def _extract_bricks(self, frame: np.ndarray) -> np.ndarray:
        """Extract brick state from the frame."""
        bricks_zone = frame[self.BRICKS_Y_START:self.BRICKS_Y_END, :]
        brick_mask = bricks_zone > 0
        
        # Reshape into grid of layers and bricks
        reshaped_mask = brick_mask.reshape(
            self.NUM_BRICK_LAYERS,
            self.brick_y_length,
            self.NUM_BRICKS_PER_LAYER,
            self.brick_x_length,
        )
        
        # Check if any pixel in each brick is active
        return reshaped_mask.all(axis=(1, 3))
    
    def _generate_object_vector_features(self, frame: np.ndarray, paddle_x_norm: float, paddle_y_norm: float, paddle_dx_norm: float, ball_x_norm: float, ball_y_norm: float, ball_dx_norm: float, ball_dy_norm: float, bricks: np.ndarray, env_idx: int) -> np.ndarray:
        """Generate object vector features for deep sets."""
        # Player vector: [x, y, dx, dy, is_active, is_player, is_ball, is_brick]
        player_vector = np.array([
            paddle_x_norm, paddle_y_norm, paddle_dx_norm, 0, 1, 1, 0, 0
        ])
        
        # Ball vector
        ball_vector = np.array([
            ball_x_norm, ball_y_norm, ball_dx_norm, ball_dy_norm,
            1 if not self._is_ball_missing(env_idx) else 0, 0, 1, 0
        ])
        
        # Brick vectors
        bricks_vectors = self._create_brick_vectors(frame, bricks, object_type="vectors")
        
        # Concatenate all vectors
        return np.concatenate([
            player_vector.reshape(1, -1),
            ball_vector.reshape(1, -1),
            bricks_vectors.reshape(-1, len(player_vector)),
        ], axis=0)
    
    def _generate_discovery_features(self, frame: np.ndarray, paddle_x_norm: float, paddle_y_norm: float, paddle_dx_norm: float, ball_x_norm: float, ball_y_norm: float, ball_dx_norm: float, ball_dy_norm: float, bricks: np.ndarray, env_idx: int) -> np.ndarray:
        """Generate discovery features with object dimensions."""
        # Player vector: [x, y, dx, dy, w, h]
        player_vector = np.array([
            paddle_x_norm, paddle_y_norm, paddle_dx_norm, 0,
            self.normalize(self.PADDLE_WIDTH, self.area_scale, True),
            self.normalize(self.PADDLE_HEIGHT, self.area_scale, True),
        ])
        
        # Ball vector (only if active)
        if not self._is_ball_missing(env_idx):
            ball_vector = np.array([
                ball_x_norm, ball_y_norm, ball_dx_norm, ball_dy_norm,
                self.normalize(self.BALL_WIDTH, self.area_scale, True),
                self.normalize(self.BALL_HEIGHT, self.area_scale, True),
            ])
        else:
            ball_vector = np.zeros(6, dtype=np.float32)
        
        # Brick vectors
        bricks_vectors = self._create_brick_vectors(frame, bricks, object_type="discovery")
        
        # Concatenate all vectors
        return np.concatenate([
            player_vector.reshape(1, -1),
            ball_vector.reshape(1, -1),
            bricks_vectors.reshape(-1, len(player_vector)),
        ], axis=0)
    
    def _create_brick_vectors(self, frame: np.ndarray, bricks: np.ndarray, object_type: str) -> np.ndarray:
        """Create feature vectors for bricks."""
        # Create coordinate grids
        j_indices, k_indices = np.meshgrid(
            np.arange(self.NUM_BRICK_LAYERS),
            np.arange(self.NUM_BRICKS_PER_LAYER),
            indexing="ij",
        )
        
        if object_type == "vectors":
            # Format: [x, y, dx, dy, is_active, is_player, is_ball, is_brick]
            bricks_vectors = np.zeros((self.NUM_BRICK_LAYERS, self.NUM_BRICKS_PER_LAYER, 8))
            bricks_vectors[:, :, 0] = self.normalize(k_indices * self.brick_x_length, frame.shape[1], True)  # x
            bricks_vectors[:, :, 1] = self.normalize(j_indices * self.brick_y_length, frame.shape[0], True)  # y
            bricks_vectors[:, :, 4] = bricks.astype(np.float32)  # is_active
            bricks_vectors[:, :, 7] = 1  # is_brick
            
        elif object_type == "discovery":
            # Format: [x, y, dx, dy, w, h]
            bricks_vectors = np.zeros((self.NUM_BRICK_LAYERS, self.NUM_BRICKS_PER_LAYER, 6))
            # Only include active bricks
            bricks_vectors[bricks, 0] = self.normalize(k_indices[bricks] * self.brick_x_length, frame.shape[1], True)  # x
            bricks_vectors[bricks, 1] = self.normalize(j_indices[bricks] * self.brick_y_length, frame.shape[0], True)  # y
            bricks_vectors[bricks, 4] = self.normalize(self.brick_x_length, self.area_scale, True)  # w
            bricks_vectors[bricks, 5] = self.normalize(self.brick_y_length, self.area_scale, True)  # h
        
        return bricks_vectors
    def _is_ball_missing(self, env_idx: int) -> bool:
        """
        Check if the ball is missing/not detected in the given environment.
        
        Args:
            env_idx: Index of the environment.
            
        Returns:
            bool: True if the ball is missing, False otherwise.
        """
        return (self.ball_x[env_idx] == 0.0 and self.ball_y[env_idx] == 0.0 and 
                self.ball_dx[env_idx] == 0.0 and self.ball_dy[env_idx] == 0.0)
    
    @staticmethod
    def normalize(x: float, scale: float, centering: bool = False) -> float:
        """
        Normalize a value to a specific range.
        
        Args:
            x: Value to normalize
            scale: Scale factor for normalization
            centering: If True, centers the value around 0 (range [-1, 1])
                      If False, scales to [0, 1] range
                      
        Returns:
            float: Normalized value
        """
        if centering:
            return (x / scale) * 2 - 1
        else:
            return x / scale
