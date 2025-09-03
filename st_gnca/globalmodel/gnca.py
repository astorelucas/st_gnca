import torch
import torch.nn as nn


class GraphCellularAutomata(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()
    self.graph = kwargs.get('graph', None)
    self.cell_model = kwargs.get('cell_model', None)
    self.device = kwargs.get('device', torch.device('cpu'))
    self.dtype = kwargs.get('dtype', torch.float32)

    def call_model(self, X_batch):
      outputs = []
      for sensor in self.graph.nodes:
        batch_sensor = batch_selector(sensor, X_batch)
        y_pred = self.cell_model(batch_sensor)
        outputs.append(y_pred)
        return torch.stack(outputs).squeeze()

    def batch_selector(sensor, X_batch):
      X_batch_new = X_batch[:, sensor, :]

      for neighbor in self.graph.neighbors(sensor):
        X_batch_new = torch.cat((X_batch_new, X_batch[:, neighbor, :]), dim=1)
        X_batch_new = X_batch_new.unsqueeze(1)

      return X_batch_new