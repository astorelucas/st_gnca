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

    self.gat_layer = GATConv(
                    in_channels=1,
                    out_channels=self.hidden_dim,
                    dropout=self.dropout,
                    heads=self.heads,
                ).to(dtype=self.dtype)
    
    self.edge_index = self.cell_model.edge_index

  def _gat_spatial_embedder(self, xt_filtered):
        sequence_out = []
        for t in range(xt_filtered.size(1)):
            #extract time step t
            xt = xt_filtered[:, t, :]
            # print(f"Time step {t}, xt shape before GAT: {xt.shape}") #torch.Size([32, 9])

            # Apply GAT layer
            xt = xt.unsqueeze(-1)
            xt_flattened = xt.contiguous().view(-1, 1)
            gat_out = self.gat_layer(xt_flattened, self.edge_index)
            gat_out = gat_out.view(xt.size(0), -1, self.gat_layer.out_channels)

            sequence_out.append(gat_out)

        spatial_embedder = torch.stack(sequence_out, dim=1)
        # print(f"Sequence out shape after GAT: {spatial_embedder.shape}") #torch.Size([32, 10, 5, 64])
        return spatial_embedder

  def call_model(self, X_batch, **kwargs):
    outputs = []

    # Store the batch graph for neighborhood extraction during forward pass
    self.cell_model.X_batch_graph = X_batch

    mode = kwargs.get('mode', 'train')
    if mode == 'train':
      self.cell_model.train()
    else:
      self.cell_model.eval()

    xt_filtered = X_batch[:, :, self.temp_dim:]  # Filter out temporal features
    spatial_embedder = self._gat_spatial_embedder(xt_filtered)

    # Pass through each sensor/node in the graph, to form the predicted output for each node
    for sensor in self.graph.nodes:
  
      y_pred = self.cell_model(X_batch, sensor, spatial_embedder)
      # print(f"Predicted shape for sensor {sensor}: {y_pred.shape}")
      outputs.append(y_pred)
    stacked_outputs = torch.stack(outputs)
    # print(f"Stacked outputs shape before permute: {stacked_outputs.shape}")
    # Stacked outputs shape before permute: torch.Size([5, 32, 3])
    final_output = stacked_outputs.permute(1, 0, 2).squeeze(2)
    return final_output
