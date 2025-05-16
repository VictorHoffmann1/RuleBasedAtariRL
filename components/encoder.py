import numpy as np
import cv2

class RuleBasedEncoder:
    """Rule-based encoder for Atari games."""
    def __init__(self,
                 num_brick_layers=6,
                 num_bricks_per_layer=18,
                 bricks_y_zone=(26,62),
                 frame_x_size=160,
                 ):
        bricks_shape = (bricks_y_zone[1] - bricks_y_zone[0], frame_x_size)
        self.num_brick_layers = num_brick_layers
        self.num_bricks_per_layer = num_bricks_per_layer
        self.brick_x_length = bricks_shape[1] // num_bricks_per_layer
        self.brick_y_length = bricks_shape[0] // num_brick_layers
        self.bricks_y_zone = bricks_y_zone
        self.frame_count = 0
        self.reset()

    def reset(self):
        """Resets the encoder state."""
        self.init = True
        self.ball_x, self.ball_y = None, None
        self.ball_dx, self.ball_dy = None, None

    def __call__(self, state):
        """Encodes the frame into a feature space. 
        Args:
            frame (np.ndarray): The input frame to encode.
        Returns:
            np.ndarray: The encoded feature space.
            (Player position, Ball position, Ball Speed, Bricks)
        """
        # Crop the frame
        frame = state[31:,8:-8]

        # Get the ball and player information (they both have the same color)
        color_mask = (frame[:,:,0] > 195) & (frame[:,:,1] < 80)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(color_mask.astype(np.uint8))
        ball_found = False
        bricks_zone = frame[self.bricks_y_zone[0]:self.bricks_y_zone[1], :, :]  # Get the bricks zone

        for i in range(1, num_labels):  # skip label 0 (background)

            x, y, _, _, area = stats[i]

            # Detect the player
            if 31 < area < 80:
                player_x = x / frame.shape[0] # No need to get the y position
            
            # Detect the ball
            elif 1 < area < 15:
                ball_found = True
                if self.init:
                    self.ball_dx, self.ball_dy = 0, 0
                    self.init = False
                else:
                    self.ball_dx, self.ball_dy = x / frame.shape[1] - self.ball_x, y / frame.shape[0] - self.ball_y
                self.ball_x, self.ball_y = x / frame.shape[1], y / frame.shape[0]
        self.frame_count += 1

        # Edge case where the ball is in contact with the player or a brick from the last layer
        if not ball_found:
            if self.ball_x is None or self.ball_y is None:
                self.ball_x, self.ball_y = 0, 0
                self.ball_dx, self.ball_dy = 0, 0
            self.ball_x += self.ball_dx / frame.shape[1]
            self.ball_y += self.ball_dy / frame.shape[0]

        # Get the bricks information (2D boolean array)
        brick_mask = (bricks_zone[:, :, 0] > 0) & (bricks_zone[:, :, 1] > 0) & (bricks_zone[:, :, 2] > 0)
        # Reshape the brick_mask into a grid of layers and bricks
        reshaped_mask = brick_mask.reshape(self.num_brick_layers, self.brick_y_length, self.num_bricks_per_layer, self.brick_x_length)
        # Use NumPy's any() along the appropriate axes to determine if any pixel in each brick is True
        bricks = reshaped_mask.any(axis=(1, 3))

        # Concatenate the player position, ball position, ball speed, and bricks into a single feature space
        feature_space = np.concatenate((
            np.array([player_x, self.ball_x, self.ball_y, self.ball_dx, self.ball_dy]),
            bricks.flatten()
            ))

        return feature_space
