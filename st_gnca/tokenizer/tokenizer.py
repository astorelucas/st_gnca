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

  def forward(
        self,
        raw_features: torch.Tensor,
        gat_features: torch.Tensor,
        temporal_features: torch.Tensor,
        target_sensor_idx: int
    ):
        """
        Finds a target sensor and its neighbors, then aggregates and pads raw, temporal,
        and GAT features into a single tensor for that neighborhood.
        """
        # 0. Initialize accumulators for neighbor features and weights , too many indices for tensor of dimension 3
        sum_neighbor_raw = torch.zeros_like(raw_features[:, :, 0]).unsqueeze(-1) # [B, T, 1]
        # print(f"sum_neighbor_raw shape: {sum_neighbor_raw.shape}")

        sum_neighbor_gat = torch.zeros_like(gat_features[:, :, 0]) # [B, T, H]
        # print(f"sum_neighbor_gat shape: {sum_neighbor_gat.shape}")
        total_weight = 0.0

        # --- 1. Identify the nodes in the neighborhood---
        neighbors = list(self.graph.neighbors(target_sensor_idx))
        # print(f"Neighbors of sensor {target_sensor_idx}: {neighbors}")

        central_node_value = raw_features[:, :, target_sensor_idx].unsqueeze(-1) # [B, T, 1]
        # print(f"central_node_value shape: {central_node_value.shape}")
        central_node_gat = gat_features[:, :, target_sensor_idx, :] # [B, T, H]
        # print(f"central_node_gat shape: {central_node_gat.shape}")

        # --- 2. Gather features for the neighborhood ---
        for neighbor in neighbors:
            edge_weight = self.edges_weight[(target_sensor_idx, neighbor)] if (target_sensor_idx, neighbor) in self.edges_weight else self.edges_weight[(neighbor, target_sensor_idx)]
            # print(f"Edge weight between {target_sensor_idx} and {neighbor}: {edge_weight}")
            total_weight += edge_weight
            # print(f"Total weight updated: {total_weight}")
            neighbor_raw = raw_features[:, :, neighbor] * edge_weight
            # print(f"neighbor_raw shape: {neighbor_raw.shape}")
            neighbor_gat = gat_features[:, :, neighbor, :] * edge_weight
            # print(f"neighbor_gat shape: {neighbor_gat.shape}")

            sum_neighbor_raw += neighbor_raw.unsqueeze(-1) # [B, T, 1]
            # print(f"sum_neighbor_raw updated shape: {sum_neighbor_raw.shape}")
            sum_neighbor_gat += neighbor_gat # [B, T, H]
            # print(f"sum_neighbor_gat updated shape: {sum_neighbor_gat.shape}")

        # --- 3. Take the average of the neighbors' features ---
        if total_weight > 0:
            avg_neighbor_raw = sum_neighbor_raw / total_weight
            # print(f"avg_neighbor_raw shape: {avg_neighbor_raw.shape}")
            avg_neighbor_gat = sum_neighbor_gat / total_weight
            # print(f"avg_neighbor_gat shape: {avg_neighbor_gat.shape}")
        else:
            avg_neighbor_raw = torch.zeros_like(raw_features[:, :, 0])
            avg_neighbor_gat = torch.zeros_like(gat_features[:, :, 0, :])

        # print(f"avg_neighbor_raw final shape: {avg_neighbor_raw.shape}")
        concatenated_tensor = torch.cat(
            (temporal_features, 
             avg_neighbor_raw, 
             avg_neighbor_gat,
             central_node_value,
             central_node_gat,
            ),
            dim=-1)
        # print(f"Concatenated features shape: {concatenated_tensor.shape}") # (B, S, 4 + 1 + H + 1 + H)

        return concatenated_tensor
  
  def to(self, device, dtype):
        self.device = device
        self.dtype = dtype
        return self
