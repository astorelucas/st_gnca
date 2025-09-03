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

    self.cell_model.X_batch_graph = X_batch

    # Pass through each sensor/node in the graph, to form the predicted output for each node
    for sensor in self.graph.nodes:
      # batch_sensor = self.batch_selector(sensor, X_batch)
      # print(f"Sensor {sensor}, Batch sensor shape: {batch_sensor.shape}")
      y_pred = self.cell_model(X_batch, sensor)
      outputs.append(y_pred)
      return torch.stack(outputs).squeeze()

  # def batch_selector(self, sensor, X_batch):
  #   selected_tensors = []

  #   time_embedding = X_batch[:, :, 0:4] # 4 = temporal_embedding_dim
  #   selected_tensors.append(time_embedding)
  #   # print(f"Time embedding shape: {time_embedding.shape}")

  #   target_sensor_data = X_batch[:, :, sensor].unsqueeze(2)
  #   selected_tensors.append(target_sensor_data)
  #   # print(f"Sensor {sensor}, Initial shape: {target_sensor_data.shape}")

  #   for neighbor in self.graph.neighbors(sensor):
  #     neighbor_data = X_batch[:, :, neighbor].unsqueeze(2)
  #     # print(f"  Neighbor {neighbor}, shape: {neighbor_data.shape}")
  #     selected_tensors.append(neighbor_data)

  #   combined_batch = torch.cat(selected_tensors, dim=2)
  #   return combined_batch