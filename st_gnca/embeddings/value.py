import torch
from torch import nn
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_scaler(train_ds, device="cuda", sample_size=100_000):
    """Optimized scaler initialization with random sampling"""
    # 1. Estimate min/max from a random sample (no full dataset iteration)
    sample_indices = np.random.choice(len(train_ds), size=min(sample_size, len(train_ds)), replace=False)
    
    # 2. Vectorized extraction using dataset.__getitem__
    sample_ys = []
    for idx in sample_indices:
        _, y = train_ds[idx]  # Direct access avoids DataLoader overhead
        sample_ys.append(y.cpu().numpy() if torch.is_tensor(y) else y)
    
    # 3. Stack and compute stats in one batch
    scaler = ScalingTransform(np.stack(sample_ys), device=device)
    
    print(f"Scaler initialized with {len(sample_ys)} samples (min={scaler.min:.4f}, max={scaler.max:.4f})")
    return scaler

class ValueEmbedding(nn.Module):
  def __init__(self, data, **kwargs):
    super().__init__()
    self.device = kwargs.get('device',DEVICE)
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
        self.device = kwargs.get('device', 'cpu')
        self.dtype = kwargs.get('dtype', torch.float32)
        self.epsilon = kwargs.get('epsilon', 1e-8)
        
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=self.dtype, device=self.device)
        
        # Safely handle NaN/Inf
        data = torch.nan_to_num(
            data, 
            nan=0.0, 
            posinf=torch.finfo(self.dtype).max, 
            neginf=torch.finfo(self.dtype).min
        )
        
        self.register_buffer('min', torch.min(data))
        self.register_buffer('range', (torch.max(data) - self.min) + self.epsilon)

    def forward(self, x):
        return (x - self.min) / self.range

    def denormalize(self, x):
        return x * self.range + self.min

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        if len(args) > 0:
            if isinstance(args[0], torch.dtype):
                self.dtype = args[0]
            elif isinstance(args[0], (str, torch.device)):
                self.device = args[0] if isinstance(args[0], str) else args[0].type
        return self
  

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