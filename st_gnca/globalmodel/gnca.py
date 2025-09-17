import torch
import torch.nn as nn


class GraphCellularAutomata(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()
    self.graph = kwargs.get('graph', None)
    self.cell_model = kwargs.get('cell_model', None)
    self.device = kwargs.get('device', torch.device('cpu'))
    self.dtype = kwargs.get('dtype', torch.float32)

  def call_model(self, X_batch, **kwargs):
    outputs = []

    # Store the batch graph for neighborhood extraction during forward pass
    self.cell_model.X_batch_graph = X_batch

    mode = kwargs.get('mode', 'train')
    if mode == 'train':
      self.cell_model.train()
    else:
      self.cell_model.eval()

    # Pass through each sensor/node in the graph, to form the predicted output for each node
    for sensor in self.graph.nodes:
      y_pred = self.cell_model(X_batch, sensor)
      # print(f"Predicted shape for sensor {sensor}: {y_pred.shape}")
      outputs.append(y_pred)
    stacked_outputs = torch.stack(outputs)
    # print(f"Stacked outputs shape before permute: {stacked_outputs.shape}")
    # Stacked outputs shape before permute: torch.Size([5, 32, 3])
    final_output = stacked_outputs.permute(1, 0, 2).squeeze(2)
    return final_output
