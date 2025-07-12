import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, pooling="mean"):
        super().__init__()
        self.pooling = pooling

        self.conv1 = GCNConv(
            input_dim - 3, hidden_dim
        )  # Exclude is_player, is_ball, is_brick
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, num_nodes, 8)
        Assumes last 3 dims are is_player, is_ball, is_brick and are NOT used for GNN input
        """
        x, mask = self.trim(x)  # Remove zero-padded objects
        batch_size, num_nodes, feat_dim = x.shape
        device = x.device

        # Extract core features for all batches at once
        core_features = x[:, :, :5]  # (B, N, 5) - [x, y, dx, dy, is_active]

        # Vectorized object type identification
        is_ball = x[:, :, 6] == 1  # (B, N)
        is_player = x[:, :, 5] == 1  # (B, N)
        is_brick = x[:, :, 7] == 1  # (B, N)

        # Find indices for each batch
        ball_indices = is_ball.nonzero(
            as_tuple=False
        )  # (num_balls, 2) - [batch_idx, node_idx]
        player_indices = is_player.nonzero(as_tuple=False)
        brick_indices = is_brick.nonzero(as_tuple=False)

        # Validate one ball and paddle per batch
        assert ball_indices.size(0) == batch_size, (
            f"Expected {batch_size} balls, got {ball_indices.size(0)}"
        )
        assert player_indices.size(0) == batch_size, (
            f"Expected {batch_size} paddles, got {player_indices.size(0)}"
        )

        # Build edge index efficiently
        all_edges = []
        batch_assignment = []
        node_offset = 0

        for b in range(batch_size):
            # Get node indices for this batch
            ball_node = ball_indices[b, 1].item()
            player_node = player_indices[b, 1].item()
            brick_nodes = brick_indices[
                brick_indices[:, 0] == b, 1
            ]  # All bricks in batch b

            # Add node offset for global indexing
            ball_global = ball_node + node_offset
            player_global = player_node + node_offset
            brick_globals = brick_nodes + node_offset

            # Ball <-> Player edges
            edges = [[ball_global, player_global], [player_global, ball_global]]

            # Ball <-> Brick edges
            for brick_global in brick_globals:
                edges.extend(
                    [
                        [ball_global, brick_global.item()],
                        [brick_global.item(), ball_global],
                    ]
                )

            all_edges.extend(edges)

            # Batch assignment for each node
            batch_assignment.extend([b] * num_nodes)
            node_offset += num_nodes

        # Convert to tensors
        if all_edges:
            edge_index = (
                torch.tensor(all_edges, dtype=torch.long, device=device)
                .t()
                .contiguous()
            )
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

        batch_tensor = torch.tensor(batch_assignment, dtype=torch.long, device=device)

        # Flatten features for batched processing
        x_flat = core_features.view(-1, 5)  # (B*N, 5)

        # GNN forward pass
        h = F.relu(self.conv1(x_flat, edge_index))
        h = self.conv2(h, edge_index)

        # Pooling over nodes to get graph-level feature
        if self.pooling == "mean":
            out = global_mean_pool(h, batch_tensor)
        elif self.pooling == "max":
            out = global_max_pool(h, batch_tensor)
        elif self.pooling == "sum":
            out = global_add_pool(h, batch_tensor)
        else:
            raise ValueError(f"Unsupported pooling: {self.pooling}")
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


class GNNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, n_features, hidden_dim, output_dim, pooling):
        super().__init__(observation_space, features_dim=output_dim)
        self.gnn_encoder = GNN(n_features, hidden_dim, output_dim, pooling)

    def forward(self, observations):
        return self.gnn_encoder(observations)


class CustomGNNPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        n_features=8,
        hidden_dim=64,
        output_dim=32,
        pooling="max",
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=GNNFeaturesExtractor,
            features_extractor_kwargs=dict(
                n_features=n_features,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                pooling=pooling,
            ),
            **kwargs,
        )
