import torch
import torch.nn as nn
import torch.nn.functional as F


class RepresentationNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, top_k=64):
        super().__init__()
        # Self-Interaction MLP
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Pairwise interaction MLP (excluding self-interaction)
        self.xi = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Top-K attention mechanism to select important interactions
        self.top_k_attention = TopKAttention(
            input_dim=input_dim,  # Input dimension for attention
            proj_dim=hidden_dim,  # Projection dimension for attention
            top_k=top_k,  # Number of top interactions to select
        )

        # Global MLP after pooling
        self.rho = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                hidden_dim,  # Output dimension of the global MLP
                hidden_dim,
            ),
        )
        self.pairwise_auxiliary_mlp = nn.Linear(hidden_dim, 2)
        self.interaction_feat = None  # Store interaction features for auxiliary loss
        self.x_i, self.x_j = None, None  # Initialize to None for later use
        self.auxiliary_criterion = nn.BCEWithLogitsLoss()

        self.hidden_dim = hidden_dim
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=torch.sqrt(torch.tensor(2.0)))
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x, obj_padding_mask = self.trim(x)  # Remove zero-padded objects
        B = x.shape[0]

        # Get top-k interactions
        ij_idxs, ij_weights = self.top_k_attention(x, padding_mask=obj_padding_mask)

        top_k = ij_idxs.shape[1]  # Number of top-k interactions

        # Efficient self-mask computation
        self_mask = ij_idxs[..., 0] == ij_idxs[..., 1]  # (B, top_k)
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, top_k)

        # Vectorized indexing
        i_idx, j_idx = ij_idxs[..., 0], ij_idxs[..., 1]
        self.x_i = x[batch_indices, i_idx]  # (B, top_k, 6)
        self.x_j = x[batch_indices, j_idx]  # (B, top_k, 6)

        # Parallel feature computation
        feat_self = self.phi(self.x_i)  # (B, top_k, H)
        feat_pair = self.xi(torch.cat([self.x_i, self.x_j], dim=-1))  # (B, top_k, H)

        # Efficient conditional selection
        self.interaction_feat = torch.where(
            self_mask.unsqueeze(-1), feat_self, feat_pair
        )  # (B, top_k, H)

        # Optimized weighted pooling
        pooled = torch.sum(
            self.interaction_feat * ij_weights.unsqueeze(-1), dim=1
        )  # (B, H)

        return self.rho(pooled)

    def pairwise_auxiliary_loss(self):
        """
        Compute auxiliary loss based on pairwise interactions.
        This can be used to encourage the model to learn meaningful pairwise relationships.
        """
        if self.x_i is None or self.x_j is None:
            return torch.zeros(1, device=self.x_i.device), torch.zeros(
                1, device=self.x_i.device
            )

        auxiliary_preds = self.pairwise_auxiliary_mlp(self.interaction_feat)
        is_collision_pred, is_closer_pred = (
            auxiliary_preds[..., 0],
            auxiliary_preds[..., 1],
        )

        denormalized_x_i = self.denormalize(self.x_i)
        denormalized_x_j = self.denormalize(self.x_j)

        is_collision_target = self.is_collision(denormalized_x_i, denormalized_x_j)
        is_closer_target = self.is_closer(denormalized_x_i, denormalized_x_j)

        return self.auxiliary_criterion(
            is_collision_pred, is_collision_target.float()
        ), self.auxiliary_criterion(is_closer_pred, is_closer_target.float())

    @staticmethod
    def denormalize(x):
        # TODO: Do this properly
        out = x.clone()
        out[..., 2] = out[..., 2] * 8.0 / 160
        out[..., 3] = out[..., 3] * 8.0 / 210
        return out

    @staticmethod
    def trim(x):
        """
        Remove trailing zero-padded objects from the input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, D) with zero-padded objects.
        Returns:
            torch.Tensor: Tensor with zero-padded objects trimmed, shape (B, max_valid_N, D).
        """

        obj_padding_mask = x.abs().sum(dim=-1) != 0  # (B, N)

        max_valid = obj_padding_mask.sum(dim=1).max()

        return x[:, :max_valid, :], obj_padding_mask[
            :, :max_valid
        ]  # Return trimmed tensor and mask

    @staticmethod
    def will_collide(obj1, obj2):
        """
        Check if two objects will collide based on their positions, shapes and speeds.
        Args:
            obj1 (torch.Tensor): Position tensor of shape (..., D).
            obj2 (torch.Tensor): Position tensor of shape (..., D).
        Returns:
            torch.Tensor: Boolean tensor indicating future collisions, shape (...).
        """

        def compute_time_interval(x1, x2):
            """
            Computes the time interval during which projections overlap on one axis.
            x1 (torch.Tensor): First object's position, speed and length on one axis.
            x2 (torch.Tensor): Second object's position, speed and length on one axis.
            .
            Returns (t_enter, t_exit)
            """

            p1, v1, s1 = x1[..., 0], x1[..., 1], x1[..., 2]
            p2, v2, s2 = x2[..., 0], x2[..., 1], x2[..., 2]

            rel_v = v2 - v1
            t1 = (p1 - p2 - s2 / 2) / rel_v
            t2 = (p1 + s1 / 2 - p2) / rel_v

            # Where rel_v == 0, check if overlapping initially
            zero_mask = rel_v == 0

            # Check if objects are initially overlapping when no relative movement
            no_overlap = (p1 + s1 <= p2) | (p2 + s2 <= p1)

            t_enter = torch.min(t1, t2)
            t_exit = torch.max(t1, t2)

            # Apply the new logic for zero relative velocity
            t_enter = torch.where(
                zero_mask & no_overlap,
                torch.full_like(t_enter, float("inf")),
                torch.where(
                    zero_mask & ~no_overlap,
                    torch.full_like(t_enter, float("-inf")),
                    t_enter,
                ),
            )
            t_exit = torch.where(
                zero_mask & no_overlap,
                torch.full_like(t_exit, float("-inf")),
                torch.where(
                    zero_mask & ~no_overlap,
                    torch.full_like(t_exit, float("inf")),
                    t_exit,
                ),
            )
            return t_enter, t_exit

        t_enter_x, t_exit_x = compute_time_interval(obj1[..., ::2], obj2[..., ::2])
        t_enter_y, t_exit_y = compute_time_interval(obj1[..., 1::2], obj2[..., 1::2])

        t_enter = torch.max(t_enter_x, t_enter_y)
        t_exit = torch.min(t_exit_x, t_exit_y)

        return (t_enter <= t_exit) & (t_exit >= 0)

    @staticmethod
    def is_closer(obj1, obj2):
        obj1_pos = obj1[..., :2]  # (B, 2)
        obj2_pos = obj2[..., :2]  # (B, 2)
        obj1_next_pos = obj1_pos + obj1[..., 2:4]  # (B, 2)
        obj2_next_pos = obj2_pos + obj2[..., 2:4]  # (B, 2)

        return torch.norm(obj1_next_pos - obj2_next_pos, dim=-1) < torch.norm(
            obj1_pos - obj2_pos, dim=-1
        )

    @staticmethod
    def is_collision(x, y, eps=1.0):
        """
        Check if two objects collide based on their positions, shapes and speeds.
        Args:
            x (torch.Tensor): Position tensor of shape (B, top_k, D).
            y (torch.Tensor): Position tensor of shape (B, top_k, D).
        Returns:
            torch.Tensor: Boolean tensor indicating collisions, shape (B, top_k).
        """
        # Get positions, shapes and speeds
        x_pos, x_speed, x_shape = x[..., :2], x[..., 2:4], x[..., 4:6]
        y_pos, y_speed, y_shape = y[..., :2], y[..., 2:4], y[..., 4:6]

        # Check if the objects collide / overlap
        cond1 = torch.abs(x_pos - y_pos) < (x_shape + y_shape) / 2

        # Check if there will be a collision in the next step, since there can be object detection errors
        # when two objects touch, they can be detected as one single object
        cond2 = (
            torch.abs(x_pos + x_speed - y_pos - y_speed) < (x_shape + y_shape) / 2 + eps
        )

        return (cond1[..., 0] & cond1[..., 1]) | (cond2[..., 0] & cond2[..., 1])


class TopKAttention(nn.Module):
    """
    Top-K attention mechanism to select important interactions.
    This module computes pairwise attention scores and selects the top-k pairs.
    """

    def __init__(self, input_dim, proj_dim, top_k, verbose=False):
        """
        Initialize the Top-K attention mechanism.
        """
        super().__init__()
        self.top_k = top_k
        self.scale = proj_dim**-0.5

        self.query_proj = nn.Linear(input_dim, proj_dim)  # Query projection
        self.key_proj = nn.Linear(input_dim, proj_dim)  # Key projection

        self.verbose = verbose
        if verbose:
            self.count = 0

    def forward(self, x, padding_mask=None):
        B, L, _ = x.shape

        Q = self.query_proj(x)  # (B, L, proj_dim)
        K = self.key_proj(x)  # (B, L, proj_dim)

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Efficient masking
        if padding_mask is not None:
            # Create 2D mask efficiently
            mask_2d = padding_mask.unsqueeze(2) & padding_mask.unsqueeze(1)
            attn_scores.masked_fill_(~mask_2d, float("-inf"))

        # Flatten and get top-k
        attn_scores_flat = attn_scores.view(B, -1)
        topk_vals, topk_idx = torch.topk(
            attn_scores_flat, min(self.top_k, L**2), dim=-1, largest=True
        )
        topk_weights = F.softmax(topk_vals, dim=-1)

        # Convert indices efficiently
        row_idx = topk_idx // L
        col_idx = topk_idx % L
        topk_indices = torch.stack([row_idx, col_idx], dim=-1)

        if self.verbose:
            if self.count % 1000 == 0:  # Print every 1000 calls
                print(f"Top-{min(L**2, 10)} Pairs:")
                for i in range(min(L**2, 10)):
                    print(
                        f"{topk_indices[0, i, 0].item()} -> {topk_indices[0, i, 1].item()} with weight {topk_weights[0][i].item():.2f}"
                    )
                print(f"Number of objects: {L}")
            self.count += 1

        return topk_indices, topk_weights


# TODO: Maybe do a Dynamics Network specific for the pairwise interactions MLP


class DynamicsNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        proj_dim,
        action_space_size,
        reward_clipped=True,
    ):
        """Dynamics network: Predicts dynamics of the environment to aid learning.

        The network learns three auxiliary tasks:
        1. Action Prediction: Given previous and current states, predict the action that caused the transition
        2. State Prediction: Given previous state and action, predict the current state
        3. Reward Prediction: Given the predicted current state, predict the reward received

        Parameters
        ----------
        input_dim: int
            dimensionality of the hidden state representations
        hidden_dim: int
            hidden layer size for the prediction networks
        proj_dim: int
            projection dimension for contrastive learning
        action_space_size: int
            number of actions in the environment
        reward_clipped: bool
            whether rewards are clipped (use Tanh activation)
        reward_scale: float
            scaling factor for reward prediction loss
        lambd: float
            weight for off-diagonal terms in Barlow Twins loss
        """
        super().__init__()

        self.action_space_size = action_space_size

        # Predicts the action that caused transition from prev_state to current_state
        self.action_predictor = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_space_size),
        )

        # Predicts current state from previous state + action
        self.state_predictor = nn.Sequential(
            nn.Linear(input_dim + action_space_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )

        # Predicts reward from the current state and action
        self.reward_predictor = nn.Sequential(
            nn.Linear(input_dim + action_space_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
        self.batch_norm = nn.BatchNorm1d(proj_dim, affine=False)

    def forward(self, hidden_state, action, reward, next_state):
        # Current State Prediction: predict current state from previous state + action
        # Add action embedding to the previous hidden states
        one_hot_action = F.one_hot(action, num_classes=self.action_space_size).float()
        state_and_action = torch.cat([hidden_state, one_hot_action], dim=-1)

        # Predict current hidden state based on previous hidden state and action that was taken
        predicted_next_state = self.state_predictor(state_and_action)

        # Reward prediction: predict reward
        predicted_reward = self.reward_predictor(state_and_action).flatten()

        # Compute losses
        # State prediction: predict current state from previous state + action
        projection_loss = self.project(predicted_next_state, next_state)
        # Reward prediction: predict reward received for this transition
        reward_prediction_loss = F.mse_loss(predicted_reward, reward)

        return projection_loss, reward_prediction_loss

    def project(self, hidden_state, next_hidden_state):
        """Project hidden states to a higher space and computes cosine similarity."""
        x1 = self.projector(next_hidden_state).detach()  # (B, output_dim)
        x2 = self.predictor(self.projector(hidden_state))  # (B, output_dim)

        batch_loss = self.consist_loss_func(x1, x2)
        return batch_loss.mean()

    @staticmethod
    def consist_loss_func(f1, f2):
        """Consistency loss function: similarity loss
        Parameters
        """
        f1 = F.normalize(f1, p=2.0, dim=-1, eps=1e-5)
        f2 = F.normalize(f2, p=2.0, dim=-1, eps=1e-5)
        return -(f1 * f2).sum(dim=1)
