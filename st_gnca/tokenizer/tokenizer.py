import pandas as pd
import torch
from torch import nn
import networkx as nx

from tensordict import TensorDict
from st_gnca.modules.common import TensorDictDataframe

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeighborhoodTokenizer(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()
    self.graph = kwargs.get('graph', None)
    # self.edge_index = kwargs.get('edge_index', None)
    self.max_graph_degree = max(dict(self.graph.degree()).values())
    self.temp_dim = kwargs.get('temp_dim', 4)  # Default temporal embedding dimension
    self.hidden_dim = kwargs.get('hidden_dim', 64)
    self.dtype = kwargs.get('dtype', torch.float32)
    self.edges_weight = nx.get_edge_attributes(self.graph, 'weight')

  def aggregate_neighbor_features(self, target_sensor_idx, subset_nodes, gat_features):
        """
        Aggregates GAT features of neighbors of a target sensor using a weighted sum.

        Args:
            target_sensor_idx (int): The index of the central node.
            subset_nodes (torch.Tensor): The tensor of original node indices in the subgraph.
            gat_features (torch.Tensor): The full GAT feature tensor [B, T, N, H].
            
        Returns:
            torch.Tensor: The weighted sum of neighbor features.
        """
        sum_neighbor_gat = torch.zeros_like(gat_features[:, :, 0])
        # print(f"sum_neighbor_gat shape: {sum_neighbor_gat.shape}")
        total_weight = 0
        central_subgraph_idx = (subset_nodes == target_sensor_idx).nonzero(as_tuple=True)[0]

        # Iterate through all nodes in the k-hop subgraph
        for node_id in subset_nodes:
            # print(f"Node ID in subset_nodes: {node_id}")
            # Skip the central node itself to only aggregate neighbors
            if node_id == central_subgraph_idx:
                continue

            neighbor = node_id.item() # Convert tensor to Python int
            # print(f"Processing neighbor: {neighbor}")
            
            # Get edge weight, handling bidirectional edges
            if (target_sensor_idx, neighbor) in self.edges_weight:
                edge_weight = self.edges_weight[(target_sensor_idx, neighbor)]
            elif (neighbor, target_sensor_idx) in self.edges_weight:
                edge_weight = self.edges_weight[(neighbor, target_sensor_idx)]
            else:
                # Handle cases where there is no direct edge weight in the dictionary
                edge_weight = 0

            total_weight += edge_weight
            local_sensor_idx = (subset_nodes == node_id).nonzero(as_tuple=True)[0]
            # print(f"Local sensor index in subgraph: {local_sensor_idx}")
            # Slice the gat_features tensor using the original node ID
            neighbor_gat = (gat_features[:, :, local_sensor_idx, :] * edge_weight).squeeze(2)
            # print(f"neighbor_gat shape: {neighbor_gat.shape}")
            sum_neighbor_gat += neighbor_gat
            # print(f"Updated sum_neighbor_gat shape: {sum_neighbor_gat.shape}")

        return sum_neighbor_gat, total_weight

  def forward(
        self,
        gat_features: torch.Tensor,
        temporal_features: torch.Tensor,
        target_sensor_idx: int,
        subset_nodes = None
    ):
        """
        Finds a target sensor and its neighbors, then aggregates and pads raw, temporal,
        and GAT features into a single tensor for that neighborhood.
        """
        # 0. Initialize accumulators for neighbor features and weights , too many indices for tensor of dimension 3
        # sum_neighbor_raw = torch.zeros_like(raw_features[:, :, 0]).unsqueeze(-1) # [B, T, 1]
        # print(f"sum_neighbor_raw shape: {sum_neighbor_raw.shape}")

        sum_neighbor_gat = torch.zeros_like(gat_features[:, :, 0]) # [B, T, H]
        # print(f"sum_neighbor_gat shape: {sum_neighbor_gat.shape}")
        total_weight = 0.0

        # --- 1. Identify the nodes in the neighborhood---
        neighbors = list(self.graph.neighbors(target_sensor_idx))
        if len(neighbors) == 0:
            neighbors = [target_sensor_idx]  # If no neighbors, use the node itself
        # print(f"Neighbors of sensor {target_sensor_idx}: {neighbors}")
        # spatial_features_central = spatial_features[:, target_sensor_idx, :]
        # central_node_value = raw_features[:, :, target_sensor_idx].unsqueeze(-1)  # [B, T, 1]
        # print(f"central_node_value shape: {central_node_value.shape}")
        # print(f"spatial_features_central shape: {spatial_features_central.shape}")
        local_sensor_idx = (subset_nodes == target_sensor_idx).nonzero(as_tuple=True)[0]

        central_node_gat = gat_features[:, :, local_sensor_idx, :].squeeze(2)  # [B, T, H]
        # print(f"central_node_gat shape: {central_node_gat.shape}")

        # --- 2. Gather features for the neighborhood ---
        sum_neighbor_gat, total_weight = self.aggregate_neighbor_features(target_sensor_idx, subset_nodes, gat_features)
        # if len(neighbors) > 0:
        #     for neighbor in neighbors:
        #         edge_weight = self.edges_weight[(target_sensor_idx, neighbor)] if (target_sensor_idx, neighbor) in self.edges_weight else self.edges_weight[(neighbor, target_sensor_idx)]
        #         # print(f"Edge weight between {target_sensor_idx} and {neighbor}: {edge_weight}")
        #         total_weight += edge_weight
        #         # print(f"Total weight updated: {total_weight}")
        #         # neighbor_raw = raw_features[:, :, neighbor] * edge_weight
        #         # print(f"neighbor_raw shape: {neighbor_raw.shape}")
        #         neighbor_gat = gat_features[:, :, neighbor, :] * edge_weight
        #         # print(f"neighbor_gat shape: {neighbor_gat.shape}")
        #         # neighbor_spatial = spatial_features[:, :, neighbor, :]
        #         # print(f"neighbor_spatial shape: {neighbor_spatial.shape}")

        #         # sum_neighbor_raw += neighbor_raw.unsqueeze(-1) + neighbor_spatial.unsqueeze(-1) # [B, T, 1]
        #         # print(f"sum_neighbor_raw updated shape: {sum_neighbor_raw.shape}")
        #         sum_neighbor_gat += neighbor_gat # [B, T, H]
        #         # print(f"sum_neighbor_gat updated shape: {sum_neighbor_gat.shape}")
        # else:
        #     # if no neighbors, set sums to -1
        #     # sum_neighbor_raw = torch.ones_like(raw_features[:, :, 0]).unsqueeze(-1) * -1 # [B, T, 1]
        #     sum_neighbor_gat = torch.ones_like(gat_features[:, :, 0]) * -1 # [B, T, H]

        # --- 3. Take the average of the neighbors' features ---
        if total_weight > 0:
            # avg_neighbor_raw = sum_neighbor_raw / total_weight
            # print(f"avg_neighbor_raw shape: {avg_neighbor_raw.shape}")
            avg_neighbor_gat = sum_neighbor_gat / total_weight
            # print(f"avg_neighbor_gat shape: {avg_neighbor_gat.shape}")
        else:
            # avg_neighbor_raw = torch.zeros_like(raw_features[:, :, 0])
            avg_neighbor_gat = torch.zeros_like(gat_features[:, :, 0, :])

        # print(f"avg_neighbor_raw final shape: {avg_neighbor_raw.shape}")
        # print(f"avg_neighbor_gat final shape: {avg_neighbor_gat.shape}")
        # print(f"central_node_value final shape: {central_node_value.shape}")
        # print(f"central_node_gat final shape: {central_node_gat.shape}")
        # print(f"temporal_features shape: {temporal_features.shape}")
        concatenated_tensor = torch.cat(
            (temporal_features, #4
             avg_neighbor_gat, #64
             central_node_gat, #64
            ),
            dim=-1)
        # print(f"Concatenated features shape: {concatenated_tensor.shape}") # (B, S, 4 + 1 + H + 1 + H)

        return concatenated_tensor
  
  def to(self, device, dtype):
        self.device = device
        self.dtype = dtype
        return self
