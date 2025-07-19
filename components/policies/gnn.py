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
            input_dim, hidden_dim
        )
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, num_nodes, n_features)
        where features = [x, y, vx, vy, w, h]
        """
        batch_size, num_nodes, n_features = x.shape
        
        # Get padding mask - True for valid nodes, False for padded nodes
        mask = x.abs().sum(dim=-1) != 0  # Shape: (batch_size, num_nodes)
        
        batch_outputs = []
        
        for b in range(batch_size):
            # Get valid nodes for this batch item
            valid_mask = mask[b]  # Shape: (num_nodes,)
            valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
            
            if len(valid_indices) == 0:
                # No valid nodes, return zero tensor
                batch_outputs.append(torch.zeros(self.conv2.out_channels, device=x.device, dtype=x.dtype))
                continue
                
            # Extract valid node features
            node_features = x[b, valid_indices]  # Shape: (num_valid_nodes, n_features)
            
            # Create edge indices based on velocity features (3rd and 4th features)
            # Edge exists if either node has non-zero velocity
            edge_indices = []
            for i, idx_i in enumerate(valid_indices):
                for j, idx_j in enumerate(valid_indices):
                    if i != j:  # No self-loops
                        node_i_features = x[b, idx_i]
                        node_j_features = x[b, idx_j]
                        # Check if either node has velocity (3rd or 4th feature != 0)
                        if (node_i_features[3] != 0 or node_i_features[4] != 0 or 
                            node_j_features[3] != 0 or node_j_features[4] != 0):
                            edge_indices.append([i, j])
            
            if len(edge_indices) == 0:
                # No edges, just process nodes independently
                edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
            else:
                edge_index = torch.tensor(edge_indices, dtype=torch.long, device=x.device).t()
            
            # Apply GCN layers
            h = F.relu(self.conv1(node_features, edge_index))
            h = self.conv2(h, edge_index)
            
            # Apply global pooling
            batch_tensor = torch.zeros(h.size(0), dtype=torch.long, device=x.device)
            if self.pooling == "mean":
                graph_embedding = global_mean_pool(h, batch_tensor)
            elif self.pooling == "max":
                graph_embedding = global_max_pool(h, batch_tensor)
            elif self.pooling == "add":
                graph_embedding = global_add_pool(h, batch_tensor)
            else:
                raise ValueError(f"Unknown pooling method: {self.pooling}")
            
            batch_outputs.append(graph_embedding.squeeze(0))
        
        # Stack batch outputs
        output = torch.stack(batch_outputs, dim=0)  # Shape: (batch_size, output_dim)
        return output



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
        n_features=6,
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
