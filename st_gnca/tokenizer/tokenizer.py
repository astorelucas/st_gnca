import pandas as pd
import torch
from torch import nn

from tensordict import TensorDict
from st_gnca.common import TensorDictDataframe

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeighborhoodTokenizer(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()

    self.NULL_SYMBOL = 0

    self.device = kwargs.get('device',DEVICE)
    self.dtype = kwargs.get('dtype',torch.float32)

    self.graph = kwargs.get('graph',None)

    self.num_nodes = kwargs.get('num_nodes',None)
    self.max_length = kwargs.get('max_length',None)
    self.token_dim = kwargs.get('token_dim',None)

    self.value_embedder = kwargs.get('value_embedder',None)
    self.spatial_embedding = kwargs.get('spatial_embedding',None)
    self.temporal_embedding = kwargs.get('temporal_embedding',None)

  def embedded_data(self, data, sensor):
    if isinstance(data, pd.DataFrame):
      values = data[str(sensor)].values
    elif isinstance(data, TensorDict):
      values = data[str(sensor)]
    elif isinstance(data, TensorDictDataframe):
      values = data[str(sensor)]
                
    return self.value_embedder(torch.tensor(values, dtype=self.dtype, device=self.device))
    
  def embedded_sample(self, data, sensor, index):
    if isinstance(data, pd.DataFrame):
      value = torch.tensor(data[str(sensor)].values[index], dtype=self.dtype, device=self.device)
    elif isinstance(data, TensorDict):
      value = data[str(sensor)]
    elif isinstance(data, TensorDictDataframe):
      value = data[str(sensor), index]

    return self.value_embedder(value)
  
  def tokenize(self, timestamp, values, node):
    # print(f'Tokenizing {node} at {timestamp}')
    # print(f'Value: {values[str(node)]}')
    val = self.value_embedder(values[str(node)])

    if isinstance(timestamp, str):
      timestamp = pd.Timestamp(timestamp)

    tim_emb = self.temporal_embedding(timestamp).to(self.device)
    tokens = self.spatial_embedding[node].to(self.device)
    
    # print(f'{tokens.shape} {val.shape} {tim_emb.shape}')
    tokens = torch.hstack([tokens, val ])
    tokens = torch.hstack([tokens, tim_emb])

    m = 1

    for neighbor in self.graph.neighbors(int(node)):
      m += 1
      tokens = torch.hstack([tokens, self.spatial_embedding[neighbor]])
      # print("Neighbor key:", str(neighbor), "Keys in values:", list(values.keys()))
      tokens = torch.hstack([tokens, self.value_embedder(values[str(neighbor)])])
      tokens = torch.hstack([tokens, tim_emb])

    tokens = tokens.reshape(1, m, self.token_dim)

    tokens = torch.hstack([tokens, torch.full((1, self.max_length - m, self.token_dim), self.NULL_SYMBOL, 
                                              dtype=torch.float32, device=self.device)])
    # print(f"Tokenized {node} with {m} neighbors into shape {tokens.shape}")
    return tokens
  
  def tokenize_all(self, data, sensor):

    tmp = self.embedded_data(data, sensor)
    n = len(tmp)
    tim_emb = self.temporal_embedding.all().reshape(n,4).to(self.device)
    
    tokens = self.spatial_embedding[sensor].repeat(n,1)
    tokens = torch.hstack([tokens, tmp.reshape(n,1) ])
    tokens = torch.hstack([tokens, tim_emb])
    
    m = 1

    for neighbor in self.graph.neighbors(int(sensor)):
      m += 1
      tokens = torch.hstack([tokens, self.spatial_embedding[neighbor].repeat(n,1)])
      tokens = torch.hstack([tokens, self.embedded_data(data, neighbor).reshape(n,1) ])
      tokens = torch.hstack([tokens, tim_emb])

    tokens = torch.hstack([tokens, torch.full((n,  self.max_length - m, self.token_dim), self.NULL_SYMBOL,
                                              dtype=self.dtype, device = self.device)])
    # I want an example of a tokenized sample
    # print(f"Tokenized all for {sensor} with shape {tokens[0, :, :].shape}")
    # print(f"Tokenized all for {sensor} with first token {tokens[0, 0, :]}")
    return tokens

  def tokenize_sample(self, data, node, index):

    if isinstance(data, pd.DataFrame):
      dt = data['timestamp'][index]
    elif isinstance(data, TensorDict):
      dt = index
    elif isinstance(data, TensorDictDataframe):
      dt = data['timestamp', index]

    tim_emb = self.temporal_embedding[dt]

    tokens = self.spatial_embedding[node]
    tokens = torch.hstack([tokens, self.embedded_sample(data, node, index)])
    tokens = torch.hstack([tokens, tim_emb])

    m = 1

    for neighbor in self.graph.neighbors(node):
      m += 1
      tokens = torch.hstack([tokens, self.spatial_embedding[neighbor]])
      tokens = torch.hstack([tokens, self.embedded_sample(data, neighbor, index)])
      tokens = torch.hstack([tokens, tim_emb])

    tokens = tokens.reshape(1, m, self.token_dim)

    tokens = torch.hstack([tokens, torch.full((1, self.max_length - m, self.token_dim), self.NULL_SYMBOL,
                                              dtype=self.dtype, device = self.device)])

    return tokens.reshape(self.max_length, self.token_dim)
    
  def forward(self, data, node, sample=None, **kwargs):
     if sample is None:
       return self.tokenize_all(data, node)
     else:
       return self.tokenize_sample(data, node, sample)
  
  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]
    self.value_embedder = self.value_embedder.to(*args, **kwargs)
    self.spatial_embedding = self.spatial_embedding.to(*args, **kwargs)
    self.temporal_embedding = self.temporal_embedding.to(*args, **kwargs)
    return self
  
  def tokenize_batch(self, X_batch, device="cuda"):
      """
      Handles batches from PEMS03 DataLoader where each item is (X_dict, y_tensor)
      X_dict: {timestamp: {node_id: value}} for multiple timesteps
      y_tensor: (batch_size, output_len, num_nodes)
      """
      batch_tokens = []
      edge_indices = []
      edge_attrs = []
          
  # First collect all node values for the entire batch
      batch_values = {}
      for sample_idx, X_dict in enumerate(X_batch):
          for timestamp, node_values in X_dict.items():
              if timestamp not in batch_values:
                  batch_values[timestamp] = {}
              batch_values[timestamp].update(node_values)
      
      # Process each sample
      for sample_idx, X_dict in enumerate(X_batch):
          for timestamp, node_values in X_dict.items():
              for node_id, value in node_values.items():
                  # Prepare complete values dictionary with neighbors
                  values_dict = {str(node_id): float(value)}
                  
                  # Add all neighbor values
                  neighbors = list(self.graph.neighbors(int(node_id)))
                  for neighbor in neighbors:
                      neighbor_key = str(neighbor)
                      if timestamp in batch_values and neighbor_key in batch_values[timestamp]:
                          values_dict[neighbor_key] = float(batch_values[timestamp][neighbor_key])
                      else:
                          # Handle missing neighbor values (critical for your tokenize function)
                          values_dict[neighbor_key] = 0.0  
                  
                  # Tokenize with complete context
                  tokens = self.tokenize(
                      timestamp=timestamp,
                      values=values_dict,  # Must contain node + all neighbors
                      node=node_id
                  )
                  batch_tokens.append(tokens)
                  
                  # Build edges (if needed for your model)
                  if neighbors:
                      src = torch.zeros(len(neighbors), dtype=torch.long)
                      dst = torch.arange(1, len(neighbors)+1)
                      edge_indices.append(torch.stack([src, dst]))
                      edge_attrs.append(torch.ones(len(neighbors)))  # Or use actual edge weights

      # Stack all batch elements
      if not batch_tokens:
          raise ValueError("No tokens generated - check your input data")
      
      return (
          torch.cat(batch_tokens, dim=0).to(device),  # (total_nodes, max_length, token_dim)
          torch.cat(edge_indices, dim=1).to(device) if edge_indices else None,
          torch.cat(edge_attrs).to(device) if edge_attrs else None
      )