import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class GraphCellularAutomata(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()
    self.graph = kwargs.get('graph', None)
    self.cell_model = kwargs.get('cell_model', None)
    self.device = kwargs.get('device', torch.device('cpu'))
    self.dtype = kwargs.get('dtype', torch.float32)
    self.hidden_dim = self.cell_model.hidden_dim
    self.input_dim = self.cell_model.input_dim
    self.output_dim = self.cell_model.output_dim
    self.temp_dim = kwargs.get('temp_dim', 4)  # Default temporal embedding
    self.dropout = kwargs.get('dropout', 0.15)
    self.heads = kwargs.get('heads', 1)

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

  def _gat_spatial_embedder(self, xt_filtered):
        """
        xt_filtered: [B, T, N] where N = num_nodes (features already filtered to one channel per node)
        Applies GAT per time step and per batch on the chosen GAT device, then returns on self.device.
        """
        sequence_out = []
        B, T, N = xt_filtered.size()
        for t in range(T):
            # xt_t: [B, N, 1] on gat_device
            xt_t = xt_filtered[:, t, :].to(self.gat_device).unsqueeze(-1)

            # Apply GAT per batch to respect edge_index (no batch flattening)
            outs_b = []
            for b in range(B):
                x_b = xt_t[b]            # [N, 1]
                gat_out_b = self.gat_layer(x_b, self.edge_index)  # [N, hidden_dim * heads]
                outs_b.append(gat_out_b.unsqueeze(0))             # [1, N, H]
            gat_out = torch.cat(outs_b, dim=0).to(self.dtype)     # [B, N, H] on gat_device

            sequence_out.append(gat_out.to(self.device, non_blocking=True))

        spatial_embedder = torch.stack(sequence_out, dim=1)  # [B, T, N, H] on self.device
        return spatial_embedder

  def call_model(self, X_batch, **kwargs):
    outputs = []

    X_batch = X_batch.to(self.device)
    self.cell_model.X_batch_graph = X_batch

    mode = kwargs.get('mode', 'train')
    if mode == 'train':
      self.cell_model.train()
    else:
      self.cell_model.eval()

    xt_filtered = X_batch[:, :, self.temp_dim:]  # [B, T, N]
    spatial_embedder = self._gat_spatial_embedder(xt_filtered)

    for sensor in self.graph.nodes:
      y_pred = self.cell_model(X_batch, sensor, spatial_embedder)
      outputs.append(y_pred)

    stacked_outputs = torch.stack(outputs)         # [N, B, H_out]
    final_output = stacked_outputs.permute(1, 0, 2).squeeze(2)
    return final_output

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