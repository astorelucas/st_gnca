import torch
from torch import nn
import networkx as nx

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeighborhoodTokenizer(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()
    self.graph = kwargs.get('graph', None)
    self.max_graph_degree = max(dict(self.graph.degree()).values())
    self.temp_dim = kwargs.get('temp_dim', 4) 
    self.hidden_dim = kwargs.get('hidden_dim', 64)
    self.dtype = kwargs.get('dtype', torch.float32)
    self.edges_weight = nx.get_edge_attributes(self.graph, 'weight')

#   def aggregate_neighbor_features(self, target_sensor_idx, subset_nodes, gat_features):
#         """
#         Aggregates GAT features of neighbors of a target sensor using a weighted sum.

#         Args:
#             target_sensor_idx (int): The index of the central node.
#             subset_nodes (torch.Tensor): The tensor of original node indices in the subgraph.
#             gat_features (torch.Tensor): The full GAT feature tensor [B, T, N, H].
            
#         Returns:
#             torch.Tensor: The weighted sum of neighbor features.
#         """
#         sum_neighbor_gat = torch.zeros_like(gat_features[:, :, 0])
#         total_weight = 0
#         central_subgraph_idx = (subset_nodes == target_sensor_idx).nonzero(as_tuple=True)[0]

#         for node_id in subset_nodes:

#             if node_id == central_subgraph_idx:
#                 continue

#             neighbor = node_id.item() 
#             if (target_sensor_idx, neighbor) in self.edges_weight:
#                 edge_weight = self.edges_weight[(target_sensor_idx, neighbor)]
#             elif (neighbor, target_sensor_idx) in self.edges_weight:
#                 edge_weight = self.edges_weight[(neighbor, target_sensor_idx)]
#             else:
#                 edge_weight = 0

#             total_weight += edge_weight
#             local_sensor_idx = (subset_nodes == node_id).nonzero(as_tuple=True)[0]

#             neighbor_gat = (gat_features[:, :, local_sensor_idx, :] * edge_weight).squeeze(2)
#             sum_neighbor_gat += neighbor_gat

#         return sum_neighbor_gat, total_weight

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

        # sum_neighbor_gat = torch.zeros_like(gat_features[:, :, 0]) 
        # total_weight = 0.0

        # neighbors = list(self.graph.neighbors(target_sensor_idx))
        # if len(neighbors) == 0:
        #     neighbors = [target_sensor_idx]  
        local_sensor_idx = (subset_nodes == target_sensor_idx).nonzero(as_tuple=True)[0]

        central_node_gat = gat_features[:, :, local_sensor_idx, :].squeeze(2)  
        # sum_neighbor_gat, total_weight = self.aggregate_neighbor_features(target_sensor_idx, subset_nodes, gat_features)
      
        # if total_weight > 0:
        #     avg_neighbor_gat = sum_neighbor_gat / total_weight
        # else:
        #     avg_neighbor_gat = torch.zeros_like(gat_features[:, :, 0, :])
        
        concatenated_tensor = torch.cat(
            (temporal_features, 
            #  avg_neighbor_gat, 
             central_node_gat, 
            ),
            dim=-1)

        return concatenated_tensor
  
  def to(self, device, dtype):
        self.device = device
        self.dtype = dtype
        return self
