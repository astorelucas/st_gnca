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
    val = self.value_embedder(values[str(node)])
    
    tim_emb = self.temporal_embedding(timestamp).to(self.device)
    tokens = self.spatial_embedding[node].to(self.device)
    
    tokens = torch.hstack([tokens, val ])
    tokens = torch.hstack([tokens, tim_emb])

    m = 1

    for neighbor in self.graph.neighbors(node):
      m += 1
      tokens = torch.hstack([tokens, self.spatial_embedding[neighbor]])
      tokens = torch.hstack([tokens, self.value_embedder(values[str(neighbor)])])
      tokens = torch.hstack([tokens, tim_emb])

    tokens = tokens.reshape(1, m, self.token_dim)

    tokens = torch.hstack([tokens, torch.full((1, self.max_length - m, self.token_dim), self.NULL_SYMBOL, 
                                              dtype=torch.float32, device=self.device)])
    # print(f"Tokenized {node} with {m} neighbors into shape {tokens.shape}")
    return tokens
  
  # Create an empty sequence of tokens (filled with -1)
  # Then extracts the data for a sensor and its neighbors and fill the values
  # in the tokens, together with the spatial and temporal embeddings
  def tokenize_all(self, data, sensor):

    tmp = self.embedded_data(data, sensor)
    n = len(tmp)
    tim_emb = self.temporal_embedding.all().reshape(n,2)
    
    tokens = self.spatial_embedding[sensor].repeat(n,1)
    tokens = torch.hstack([tokens, tmp.reshape(n,1) ])
    tokens = torch.hstack([tokens, tim_emb])
    
    m = 1

    for neighbor in self.graph.neighbors(sensor):
      m += 1
      tokens = torch.hstack([tokens, self.spatial_embedding[neighbor].repeat(n,1)])
      tokens = torch.hstack([tokens, self.embedded_data(data, neighbor).reshape(n,1) ])
      tokens = torch.hstack([tokens, tim_emb])

    tokens = tokens.reshape(n, m, self.token_dim)


    tokens = torch.hstack([tokens, torch.full((n,  self.max_length - m, self.token_dim), self.NULL_SYMBOL,
                                              dtype=self.dtype, device = self.device)])

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