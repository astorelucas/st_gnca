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
    
    tim_emb = self.temporal_embedding(timestamp).to(self.device)
    tokens = self.spatial_embedding[node].to(self.device)
    
    # print(f'{tokens.shape} {val.shape} {tim_emb.shape}')
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
  
  def tokenize_batch(self, batch_data, device="cuda"):
    """
    Tokenize a batch of samples and prepare graph edges.
    
    Args:
        batch_data: List of tuples (timestamp, values_dict, node_id)
        device: Target device for tensors
    
    Returns:
        batch_tokens: Tensor of shape (B, max_length, token_dim)
        edge_index: Graph connectivity (2, num_edges)
        edge_attr: Optional edge weights (num_edges,)
    """
    batch_tokens = []
    edge_indices = []
    edge_attrs = []
    
    # 1. Tokenize all samples in batch
    for timestamp, values, node in batch_data:
        # Tokenize main node and neighbors
        tokens = self.tokenize(timestamp, values, node)  # (1, max_length, token_dim)
        batch_tokens.append(tokens)
        
        # 2. Build edges for this sample's neighborhood
        neighbors = list(self.graph.neighbors(node))
        num_neighbors = len(neighbors)
        
        # Edge indices: [main -> neighbors]
        src = torch.zeros(num_neighbors, dtype=torch.long)  # Main node is index 0
        dst = torch.arange(1, num_neighbors + 1)  # Neighbors are 1..N
        sample_edges = torch.stack([src, dst])  # (2, num_neighbors)
        
        # Optional edge attributes (e.g., inverse distance)
        sample_edge_attr = torch.ones(num_neighbors) * 0.8  # Dummy weights
        
        edge_indices.append(sample_edges)
        edge_attrs.append(sample_edge_attr)
    
    # 3. Stack batch elements
    batch_tokens = torch.cat(batch_tokens, dim=0).to(device)  # (B, max_length, token_dim)
    
    # 4. Build global edge_index (accounting for batch offsets)
    edge_index = []
    edge_attr = []
    for batch_idx, (sample_edges, sample_attr) in enumerate(zip(edge_indices, edge_attrs)):
        offset = batch_idx * self.max_length
        edge_index.append(sample_edges + offset)
        edge_attr.append(sample_attr)
    
    edge_index = torch.cat(edge_index, dim=1).to(device)  # (2, total_edges)
    edge_attr = torch.cat(edge_attrs).to(device) if edge_attrs[0] is not None else None
    
    return batch_tokens, edge_index, edge_attr