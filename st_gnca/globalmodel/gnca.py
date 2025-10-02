import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from st_gnca.embeddings.value import ZTransform
from torch_geometric.data import Data, Batch
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
    self.temp_dim = kwargs.get('temp_dim', 4)  # Default temporal embedding
    self.dropout = kwargs.get('dropout', 0.2)
    self.heads = kwargs.get('heads', 1)
    self.scaler = kwargs.get('scaler', ZTransform())
    self.laplacian_components = kwargs.get('laplacian_components', 10)

    # Use CPU for GAT when running on MPS (PyG is limited on MPS)
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
    ).to(dtype=self.dtype, device=self.gat_device)

    # ensure edge_index is long and on the same device as the GAT layer
    self.edge_index = self.cell_model.edge_index
    if not isinstance(self.edge_index, torch.Tensor):
        self.edge_index = torch.as_tensor(self.edge_index, dtype=torch.long)
    else:
        self.edge_index = self.edge_index.to(dtype=torch.long)
    self.edge_index = self.edge_index.to(self.gat_device, non_blocking=True)

  def _set_gat_device(self, main_device: torch.device):
    # Ensure GAT layer and edge_index are on the same device as the main device
    self.gat_device = main_device

  def _gat_spatial_embedder(self, xt_filtered: torch.Tensor):
      """
      xt_filtered: [B, T, N, H_in]
      Applies GAT to a batched graph without slow for loops.

      Output: [B, T, N, H_out]
      """
      B, T, N, H_in = xt_filtered.size()

      # 1. Reshape the input tensor for GAT layer processing.
      #    The GAT layer expects a flattened tensor of node features with shape [num_nodes, features].
      #    Here, num_nodes = B * T * N
      node_features = xt_filtered.view(B * T * N, H_in).to(self.gat_device)

      # 2. Replicate the edge index for the entire batch.
      #    The edge_index needs to be expanded for each (B, T) slice.
      #    This creates a large edge index tensor for the super-graph.
      batch_edge_index = self.edge_index.to(self.gat_device).repeat(1, B * T)

      # 3. Adjust the edge index values to match the flattened node features.
      #    We need to add the offset (N * t * b) to each edge index.
      offsets = torch.arange(B * T, device=self.gat_device) * N
      offsets = offsets.repeat_interleave(self.edge_index.size(1))
      batch_edge_index += offsets

      # 4. Pass the entire batch through the GAT layer.
      gat_out_all = self.gat_layer(node_features, batch_edge_index)

      # 5. Reshape the output back to the original format.
      #    The output is flattened, so we reshape it to [B, T, N, H_out].
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
      
      # --- 1. Subgraph Extraction (Done on the CPU/Host once) ---
      
      # Find the k-hop subgraph nodes and the edges within that subgraph
      # Assuming k=1 (immediate neighbors) for efficiency, 
      # but you can use a larger k based on your GAT depth.
      num_layers = self.gat_layer.num_layers if hasattr(self.gat_layer, 'num_layers') else 1
      
      # We use the unbatched edge_index for subgraph extraction.
      subset_nodes, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
          node_idx=torch.tensor([sensor_idx], dtype=torch.long),
          num_hops=num_layers,
          edge_index=self.edge_index,
          relabel_nodes=True,  # Relabels nodes from 0 to N_sub-1
          num_nodes=N         # Total number of nodes
      )
      
      N_sub = subset_nodes.size(0)  # Number of nodes in the subgraph
  

      # --- 2. Input and Edge Preparation for Batched Subgraph ---
      
      # 2a. Slice the input features (xt_filtered) to include only subgraph nodes
      # xt_sub is [B, T, N_sub, H_in]
      xt_sub = xt_filtered[:, :, subset_nodes, :]
      
      # 2b. Flatten the features to match the GAT input expectation: [B*T*N_sub, H_in]
      node_features_sub = xt_sub.reshape(B * T * N_sub, H_in).to(self.gat_device)

      # 2c. Replicate the subgraph edge index for the entire B*T batch.
      # The new total number of graphs is B * T
      batch_size_total = B * T
      sub_edge_index = sub_edge_index.to(self.gat_device)
      
      # sub_edge_index_batched is [2, E_sub * B * T]
      sub_edge_index_batched = sub_edge_index.repeat(1, batch_size_total)

      # 2d. Adjust the edge index values using offsets based on the subgraph size.
      offsets = torch.arange(batch_size_total, device=self.gat_device) * N_sub
      offsets = offsets.repeat_interleave(sub_edge_index.size(1))
      sub_edge_index_batched += offsets

      # --- 3. Run GAT on the Subgraph Batch ---
      
      # gat_out_sub is [B*T*N_sub, H_out]
      gat_out_sub = self.gat_layer(node_features_sub, sub_edge_index_batched)

      # --- 4. Reshape and Return ---
      
      # Reshape back to [B, T, N_sub, H_out]
      gat_out_sub = gat_out_sub.view(B, T, N_sub, -1)

      return gat_out_sub.to(self.device, dtype=self.dtype, non_blocking=True), subset_nodes
  
  def call_model(self, X_batch, **kwargs):
    outputs = []

    X_batch = X_batch.to(self.device)
    self.cell_model.X_batch_graph = X_batch

    self.cell_model.train(mode=(kwargs.get('mode', 'train') == 'train'))

    xt_filtered = X_batch[:, :, self.temp_dim:]  # [B, T, N] torch.Size([32, 12, 370])
    # print(f"Filtered x shape: {xt_filtered.shape}")

    # Apply linear transformation to each node's time series to sum with spatial embeddings
    x_linear = self.Linear_in(xt_filtered.unsqueeze(-1))  # [B, T, N, L] torch.Size([32, 12, 358, 10])
    # print(f"Linear x shape: {x_linear.shape}")

    spatial_embedder = self.spatial_emb.all().to(self.device, dtype=self.dtype)
    self.scaler.fit(spatial_embedder)
    spatial_embedder = self.scaler.forward(spatial_embedder)
    spatial_embedder = spatial_embedder.unsqueeze(0).repeat(xt_filtered.size(0), 1, 1)
    # print(f"Spatial embedder shape: {spatial_embedder.shape}") # torch.Size([32, 169, 10])
    # print(f"Spatial embedder example: {spatial_embedder[0, :]}")  # Example for the first node

    encoder = x_linear + spatial_embedder.unsqueeze(1)
    # print(f"encoder expanded shape: {encoder.shape}") # torch.Size([32, 12, 370, 10])

    # gat_embedder = self._gat_spatial_embedder(encoder)
    # self.scaler.fit(gat_embedder)
    # gat_embedder = self.scaler.forward(gat_embedder)
    # print(f"GAT embedder example: {gat_embedder[0, 0, 0, :]}")  # Example for the first batch, first time step, first node
    # print(f"GAT embedder shape: {gat_embedder.shape}") #torch.Size([32, 12, 370, 64])

    # print(f"Input x shape: {x.shape}") # Input x shape: torch.Size([32, 10, 9])
    x_time = xt_filtered[:, :, 0:self.temp_dim]  # Extract temporal features
    self.scaler.fit(x_time)
    x_time = self.scaler.forward(x_time)
    # print(f"Time features example:÷ {x_time[0, 0, :]}")  # Example for the first batch, first time step
    # print(f"Time features shape: {x_time.shape}") #torch.Size([32, 10, 4])

    for sensor in sorted(self.graph.nodes):
      gat_embedder, subset_nodes = self._subgat_spatial_embedder(encoder, sensor)
      self.scaler.fit(gat_embedder)
      gat_embedder = self.scaler.forward(gat_embedder)
      # print(f"GAT embedder shape for sensor {sensor}: {gat_embedder.shape}") #torch.Size([32, 12, N_sub, 64])
      y_pred = self.cell_model(sensor, gat_embedder, x_time, subset_nodes)
      outputs.append(y_pred)

    # stacked_outputs = torch.stack(outputs)         # [N, B, H_out]
    # final_output = stacked_outputs.permute(1, 0, 2).squeeze(2)
    stacked_outputs = torch.stack(outputs, dim=1)
    # print(f"Outputs shape: {stacked_outputs.shape}") # [B, N, H_out] torch.Size([32, 358, 3])
    # print("Finishing GNCA forward pass.")
    return stacked_outputs

  def to(self, device):
    # Update main device
    self.device = device
    self.cell_model.to(device)
    # Recompute best device for GAT and move layer + edge_index accordingly
    self._set_gat_device(device)
    self.gat_layer.to(self.gat_device)
    if isinstance(self.edge_index, torch.Tensor):
        self.edge_index = self.edge_index.to(self.gat_device)
    return self