import torch
from torch import nn

class ValueEmbedding(nn.Module):
  def __init__(self, data, **kwargs):
    super().__init__()
    self.device = kwargs.get('device','cpu')
    self.dtype = kwargs.get('dtype',torch.float32)
    self.type = kwargs.get('value_embedding_type','normalization')

    if self.type == 'normalization':
      self.embedder = ZTransform(data, **kwargs)
    elif self.type == 'scaling':
      self.embedder = ScalingTransform(data, **kwargs)
    else:
      raise ValueError("Unknown embedder type!")
    
  def forward(self, x):
     return self.embedder.forward(x)
  
  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]
    self.embedder = self.embedder.to(*args, **kwargs)

    return self

class ScalingTransform(nn.Module):
  def __init__(self, data, **kwargs):
    super().__init__()
    self.device = kwargs.get('device','cpu')
    self.dtype = kwargs.get('dtype',torch.float32)
    if not isinstance(data, torch.Tensor):
      data = torch.tensor(data, dtype=self.dtype, device=self.device)
    self.min = torch.min(torch.nan_to_num(data,0,0,0))
    max = torch.max(torch.nan_to_num(data,0,0,0))
    self.range = max - self.min
    
  def forward(self, x):
     return (x - self.min) / self.range
  
  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]
    self.min = self.min.to(*args, **kwargs)
    self.range = self.range.to(*args, **kwargs)

    return self
  
  import torch
from torch import nn

class ZTransform(nn.Module):
  def __init__(self, data, **kwargs):
    super().__init__()
    self.device = kwargs.get('device','cpu')
    self.dtype = kwargs.get('dtype',torch.float32)
    if not isinstance(data, torch.Tensor):
      data = torch.tensor(data, dtype=self.dtype, device=self.device)
    self.mu = torch.nanmean(data)
    self.sigma = torch.std(torch.nan_to_num(data,0,0,0))
    
  def forward(self, x):
     #return z(x, self.mu, self.sigma)
     return (x - self.mu) / self.sigma
  
  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]
    self.mu = self.mu.to(*args, **kwargs)
    self.sigma = self.sigma.to(*args, **kwargs)

    return self