import numpy as np


class RuleBasedAgent:
    def __init__(self):
        self.player_offset = 0.1
        self.no_op_threshold = 0.07

    def predict(self, features):
        """
        Perform an action based on the observation.
        Vectorized implementation using numpy.
        """
        player_x = features[:, 0] + self.player_offset
        ball_x = features[:, 1]
        ball_y = features[:, 2]

        actions = np.zeros(len(features), dtype=int)
        # Ball not in play, need to fire
        mask_fire = (ball_x == -2.0) & (ball_y == -2.0)
        actions[mask_fire] = 1
        # Player is to the left of the ball, move right
        mask_right = (player_x < ball_x - self.no_op_threshold) & (~mask_fire)
        actions[mask_right] = 2
        # Player is to the right of the ball, move left
        mask_left = (player_x > ball_x + self.no_op_threshold) & (~mask_fire)
        actions[mask_left] = 3
        # Player is aligned with the ball, do nothing (already 0)

        return actions, {}
