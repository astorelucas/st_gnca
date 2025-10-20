import torch
from torch import nn
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_scaler(train_ds, device="cuda", sample_size=100_000):
    """Optimized scaler initialization with random sampling"""
    sample_indices = np.random.choice(len(train_ds), size=min(sample_size, len(train_ds)), replace=False)
    
    sample_ys = []
    for idx in sample_indices:
        _, y = train_ds[idx]  
        sample_ys.append(y.cpu().numpy() if torch.is_tensor(y) else y)
    
    scaler = ScalingTransform(np.stack(sample_ys), device=device)
    
    print(f"Scaler initialized with {len(sample_ys)} samples (min={scaler.min:.4f}, max={scaler.max:.4f})")
    return scaler

class ValueEmbedding(nn.Module):
    def __init__(self, data, value_embedding_type, **kwargs):
        super().__init__()
        self.device = kwargs.get('device', DEVICE)
        self.dtype = kwargs.get('dtype', torch.float32)
        self.value_embedding_type = value_embedding_type

        if self.value_embedding_type == 'ztransform':
            self.embedder = ZTransform(device=self.device, dtype=self.dtype)
        elif self.value_embedding_type == 'scaling':
            self.embedder = ScalingTransform(device=self.device, dtype=self.dtype)
        elif self.value_embedding_type == 'minmax':
            self.embedder = MinMaxTransform(device=self.device, dtype=self.dtype)
        else:
            raise ValueError("Unknown embedder type!")
        
        self.embedder.fit(data)

    def forward(self, x):
        return self.embedder.forward(x)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.embedder.to(*args, **kwargs)
        
        if 'device' in kwargs:
            self.device = kwargs['device']
        elif isinstance(args[0], str):
            self.device = args[0]
        if 'dtype' in kwargs:
            self.dtype = kwargs['dtype']
        elif isinstance(args[0], torch.dtype):
            self.dtype = args[0]
        return self
    
class MinMaxTransform(nn.Module):
    def __init__(self, range=torch.tensor([0.0, 1.0]), device=DEVICE, dtype=torch.float32):
        super().__init__()
        self.min_val = None
        self.max_val = None
        self.range = range.to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype

    def forward(self, data):
        if self.min_val is None or self.max_val is None:
            raise ValueError("MinMaxTransform has not been fitted yet.")
        return (data - self.min_val) / (self.max_val - self.min_val) * (self.range[1] - self.range[0]) + self.range[0]

    def denormalize(self, x_normalized):
        """
        Reverts the Min-Max normalization.
        """
        return ((x_normalized - self.range[0]) / (self.range[1] - self.range[0])) * (self.max_val - self.min_val) + self.min_val

    def fit(self, data):
        self.min_val = data.min()
        self.max_val = data.max()

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        if isinstance(args[0], str):
            self.device = args[0]
        else:
            self.dtype = args[0]
        self.range = self.range.to(*args, **kwargs)
        return self

class ScalingTransform(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.device = kwargs.get('device', 'cpu')
        self.dtype = kwargs.get('dtype', torch.float32)
        self.epsilon = kwargs.get('epsilon', 1e-8)
        
        self.register_buffer('data_min', None)
        self.register_buffer('data_range', None)

    def fit(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=self.dtype, device=self.device)
        
        data = torch.nan_to_num(
            data, 
            nan=0.0, 
            posinf=torch.finfo(self.dtype).max, 
            neginf=torch.finfo(self.dtype).min
        )
        
        self.data_min = torch.min(data).detach()
        max_val = torch.max(data).detach()
        self.data_range = (max_val - self.data_min) + self.epsilon

    def forward(self, x):
        if self.data_min is None or self.data_range is None:
            raise RuntimeError("ScalingTransform has not been fitted yet.")
        
        return (x - self.data_min) / self.data_range

    def denormalize(self, x):
        if self.data_min is None or self.data_range is None:
            raise RuntimeError("ScalingTransform has not been fitted yet.")
        
        return x * self.data_range + self.data_min

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        
        if 'device' in kwargs:
            self.device = kwargs['device']
        elif len(args) > 0 and isinstance(args[0], (str, torch.device)):
            self.device = args[0]
            
        if 'dtype' in kwargs:
            self.dtype = kwargs['dtype']
        elif len(args) > 0 and isinstance(args[0], torch.dtype):
            self.dtype = args[0]

        return self

def nanstd(x, dim=None, keepdim=False):
    mask = ~torch.isnan(x)
    count = mask.sum(dim=dim, keepdim=True).clamp(min=1)
    masked_x = torch.where(mask, x, torch.zeros_like(x))

    mean = masked_x.sum(dim=dim, keepdim=True) / count
    var = ((torch.where(mask, x, mean) - mean) ** 2).sum(dim=dim, keepdim=True) / count
    std = torch.sqrt(var)

    if not keepdim and dim is not None:
        std = std.squeeze(dim)
    return std


class ZTransform(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.device = kwargs.get('device', 'cpu')
        self.dtype = kwargs.get('dtype', torch.float32)

        self.register_buffer('mu', torch.tensor(float('nan'), device=self.device, dtype=self.dtype))
        self.register_buffer('sigma', torch.tensor(float('nan'), device=self.device, dtype=self.dtype))

    def fit(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=self.dtype, device=self.device)
        else:
            data = data.to(self.device, self.dtype)

        dims = tuple(range(data.ndim - 1))

        if hasattr(torch, "nanmean"):
            self.mu = torch.nanmean(data, dim=dims).detach().to(self.device, self.dtype)
        else:
            self.mu = data[~torch.isnan(data)].mean(dim=dims).detach().to(self.device, self.dtype)

        if hasattr(torch, "nanstd"):
            self.sigma = torch.nanstd(data, dim=dims).detach().to(self.device, self.dtype)
        else:
            self.sigma = nanstd(data, dim=dims).detach().to(self.device, self.dtype)

        eps = torch.tensor(1e-6, device=self.device, dtype=self.dtype)
        self.sigma = torch.where(self.sigma == 0, eps, self.sigma)

    def forward(self, x):
        if torch.isnan(self.mu).any() or torch.isnan(self.sigma).any():
            raise RuntimeError("ZTransform has not been fitted yet.")
        mu = self.mu.to(x.device, x.dtype)
        sigma = self.sigma.to(x.device, x.dtype)
        return (x - mu) / sigma

    def denormalize(self, x):
        if torch.isnan(self.mu).any() or torch.isnan(self.sigma).any():
            raise RuntimeError("ZTransform has not been fitted yet.")
        mu = self.mu.to(x.device, x.dtype)
        sigma = self.sigma.to(x.device, x.dtype)
        return (x * sigma) + mu