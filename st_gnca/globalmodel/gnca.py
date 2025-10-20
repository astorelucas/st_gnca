import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from st_gnca.embeddings.value import ZTransform
from st_gnca.embeddings.spatial import SpatialEmbedding
from torch_geometric.utils import k_hop_subgraph

DEVICE = (
    torch.device('cuda') if torch.cuda.is_available()
    else torch.device('mps') if torch.backends.mps.is_available()
    else torch.device('cpu')
)

class GraphCellularAutomata(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()
    self.graph = kwargs.get('graph', None)
    self.cell_model = kwargs.get('cell_model', None)
    self.device = kwargs.get('device', DEVICE)
    self.dtype = kwargs.get('dtype', torch.float32)
    self.hidden_dim = self.cell_model.hidden_dim
    self.feature_dim = self.cell_model.feature_dim
    self.output_dim = self.cell_model.output_dim
    self.temp_dim = kwargs.get('temp_dim', 4)  
    self.dropout = kwargs.get('dropout', 0.2)
    self.heads = kwargs.get('heads', 1)
    self.scaler = kwargs.get('scaler', ZTransform())
    self.laplacian_components = kwargs.get('laplacian_components', 10)

    self._set_gat_device(self.device)

    self.spatial_emb = SpatialEmbedding(
        graph=self.graph,
        laplacian_components=self.laplacian_components,
        device=self.gat_device,
        dtype=self.dtype
    )

    self.Linear_in = nn.Linear(1, self.laplacian_components).to(dtype=self.dtype, device=self.device)

    self.gat_layer = GATConv(
        in_channels=self.laplacian_components,
        out_channels=self.hidden_dim,
        dropout=self.dropout,
        heads=self.heads,
        add_self_loops=True
    ).to(dtype=self.dtype, device=self.gat_device)

    self.edge_index = self.cell_model.edge_index
    if not isinstance(self.edge_index, torch.Tensor):
        self.edge_index = torch.as_tensor(self.edge_index, dtype=torch.long)
    else:
        self.edge_index = self.edge_index.to(dtype=torch.long)
    self.edge_index = self.edge_index.to(self.gat_device, non_blocking=True)

  def _set_gat_device(self, main_device: torch.device):
    self.gat_device = main_device

  def _gat_spatial_embedder(self, xt_filtered: torch.Tensor):
      """
      xt_filtered: [B, T, N, H_in]
      Applies GAT to a batched graph without slow for loops.

      Output: [B, T, N, H_out]
      """
      B, T, N, H_in = xt_filtered.size()

      node_features = xt_filtered.view(B * T * N, H_in).to(self.gat_device)

      batch_edge_index = self.edge_index.to(self.gat_device).repeat(1, B * T)

      offsets = torch.arange(B * T, device=self.gat_device) * N
      offsets = offsets.repeat_interleave(self.edge_index.size(1))
      batch_edge_index += offsets

      gat_out_all = self.gat_layer(node_features, batch_edge_index)

      gat_out_all = gat_out_all.view(B, T, N, -1)

      return gat_out_all.to(self.device, dtype=self.dtype, non_blocking=True)

  def _subgat_spatial_embedder(self, xt_filtered: torch.Tensor, sensor_idx: int):
      """
      Calculates the GAT output for a specific sensor and its neighbors across 
      all batches and time steps, by extracting the k-hop subgraph.

      Args:
          xt_filtered (torch.Tensor): Input tensor [B, T, N, H_in].
          sensor_idx (int): The global node index (0 to N-1) of the sensor to focus on.

      Output:
          torch.Tensor: The GAT output for the sensor and its neighbors, 
                        reshaped to [B, T, N_sub, H_out], where N_sub is 
                        the number of nodes in the subgraph.
      """
      B, T, N, H_in = xt_filtered.size()
      
      num_layers = self.gat_layer.num_layers if hasattr(self.gat_layer, 'num_layers') else 1
      
      subset_nodes, sub_edge_index, _, _ = k_hop_subgraph(
          node_idx=torch.tensor([sensor_idx], dtype=torch.long),
          num_hops=num_layers,
          edge_index=self.edge_index,
          relabel_nodes=True,  
          num_nodes=N         
      )
      
      N_sub = subset_nodes.size(0)  
  
      xt_sub = xt_filtered[:, :, subset_nodes, :]
      
      node_features_sub = xt_sub.reshape(B * T * N_sub, H_in).to(self.gat_device)

      batch_size_total = B * T
      sub_edge_index = sub_edge_index.to(self.gat_device)
      
      sub_edge_index_batched = sub_edge_index.repeat(1, batch_size_total)

      offsets = torch.arange(batch_size_total, device=self.gat_device) * N_sub
      offsets = offsets.repeat_interleave(sub_edge_index.size(1))
      sub_edge_index_batched += offsets


      gat_out_sub = self.gat_layer(node_features_sub, sub_edge_index_batched)

      gat_out_sub = gat_out_sub.view(B, T, N_sub, -1)

      return gat_out_sub.to(self.device, dtype=self.dtype, non_blocking=True), subset_nodes
  
  def call_model(self, X_batch, **kwargs):
    outputs = []

    X_batch = X_batch.to(self.device)
    self.cell_model.X_batch_graph = X_batch

    self.cell_model.train(mode=(kwargs.get('mode', 'train') == 'train'))

    xt_filtered = X_batch[:, :, self.temp_dim:]  
    x_linear = self.Linear_in(xt_filtered.unsqueeze(-1))  

    spatial_embedder = self.spatial_emb.all().to(self.device, dtype=self.dtype)
    self.scaler.fit(spatial_embedder)
    spatial_embedder = self.scaler.forward(spatial_embedder)
    spatial_embedder = spatial_embedder.unsqueeze(0).repeat(xt_filtered.size(0), 1, 1)

    encoder = x_linear + spatial_embedder.unsqueeze(1)

    x_time = X_batch[:, :, 0:self.temp_dim]  
    for sensor in sorted(self.graph.nodes):
      gat_embedder, subset_nodes = self._subgat_spatial_embedder(encoder, sensor)
      y_pred = self.cell_model(sensor, gat_embedder, x_time, subset_nodes)
      outputs.append(y_pred)

    stacked_outputs = torch.stack(outputs, dim=1)

    return stacked_outputs

  def to(self, device):
    self.device = device
    self.cell_model.to(device)
    self._set_gat_device(device)
    self.gat_layer.to(self.gat_device)
    if isinstance(self.edge_index, torch.Tensor):
        self.edge_index = self.edge_index.to(self.gat_device)
    return self