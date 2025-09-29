import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from st_gnca.embeddings.value import ZTransform

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

    # Use CPU for GAT when running on MPS (PyG is limited on MPS)
    self._set_gat_device(self.device)

    self.gat_layer = GATConv(
        in_channels=1,
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
    # If main device is MPS, place GAT on CPU to avoid MPS placeholder storage errors
    if isinstance(main_device, torch.device) and main_device.type == 'mps':
        self.gat_device = torch.device('cpu')
    else:
        self.gat_device = main_device if isinstance(main_device, torch.device) else torch.device('cpu')

  def _gat_spatial_embedder(self, xt_filtered: torch.Tensor):
      """
      xt_filtered: [B, T, N] where N = num_nodes (features already filtered to one channel per node)
      Applies GAT per time step and per batch on the chosen GAT device,
      then returns embeddings on self.device.

      Output: [B, T, N, H]
      """
      B, T, N = xt_filtered.size()

      # Flatten batch and time into one "super batch"
      xt_bt = xt_filtered.reshape(B * T, N, 1).to(self.gat_device)  # [B*T, N, 1]

      # Apply GAT per (B,T) snapshot
      outs = []
      for i in range(B * T):
          gat_out = self.gat_layer(xt_bt[i], self.edge_index)  # [N, H]
          outs.append(gat_out.unsqueeze(0))                    # [1, N, H]

      gat_out_all = torch.cat(outs, dim=0)                     # [B*T, N, H]
      gat_out_all = gat_out_all.view(B, T, N, -1)              # [B, T, N, H]

      return gat_out_all.to(self.device, dtype=self.dtype, non_blocking=True) #torch.Size([32, 12, 358, 64])

  def call_model(self, X_batch, **kwargs):
    outputs = []

    X_batch = X_batch.to(self.device)
    self.cell_model.X_batch_graph = X_batch

    self.cell_model.train(mode=(kwargs.get('mode', 'train') == 'train'))

    xt_filtered = X_batch[:, :, self.temp_dim:]  # [B, T, N] torch.Size([32, 12, 370])
    # print(f"Filtered x shape: {xt_filtered.shape}")

    spatial_embedder = self._gat_spatial_embedder(xt_filtered)
    self.scaler.fit(spatial_embedder)
    spatial_embedder = self.scaler.forward(spatial_embedder)
    # print(f"Spatial embedder example: {spatial_embedder[0, 0, 0, :]}")  # Example for the first batc÷h, first time step, first node
    # print(f"Spatial embedder shape: {spatial_embedder.shape}") #torch.Size([32, 12, 370, 64])

    # print(f"Input x shape: {x.shape}") # Input x shape: torch.Size([32, 10, 9])
    
    x_time = xt_filtered[:, :, 0:self.temp_dim]  # Extract temporal features
    self.scaler.fit(x_time)
    x_time = self.scaler.forward(x_time)
    # print(f"Time features example:÷ {x_time[0, 0, :]}")  # Example for the first batch, first time step
    # print(f"Time features shape: {x_time.shape}") #torch.Size([32, 10, 4]) 

    for sensor in self.graph.nodes:
      y_pred = self.cell_model(xt_filtered, sensor, spatial_embedder, x_time)
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