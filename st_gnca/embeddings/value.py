import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

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
        self.device = kwargs.get('device', DEVICE)
        self.dtype = kwargs.get('dtype', torch.float32)
        self.value_embedding_type = kwargs.get('value_embedding_type', 'normalization')

        if self.value_embedding_type == 'normalization':
            # Assuming ZTransform exists and has a fit method
            self.embedder = ZTransform(device=self.device, dtype=self.dtype)
        elif self.value_embedding_type == 'scaling':
            # Assuming ScalingTransform exists and has a fit method
            self.embedder = ScalingTransform(device=self.device, dtype=self.dtype)
        elif self.value_embedding_type == 'minmax':
            self.embedder = MinMaxTransform(device=self.device, dtype=self.dtype)
        else:
            raise ValueError("Unknown embedder type!")
        
        # Fit the embedder during initialization
        self.embedder.fit(data)

    def forward(self, x):
        return self.embedder.forward(x)

    def to(self, *args, **kwargs):
        # Correctly propagate the `to` call to the embedder
        super().to(*args, **kwargs)
        self.embedder.to(*args, **kwargs)
        
        # Update device and dtype attributes based on args/kwargs
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
        # The correct denormalization formula
        return ((x_normalized - self.range[0]) / (self.range[1] - self.range[0])) * (self.max_val - self.min_val) + self.min_val

    def fit(self, data):
        # Correctly assign to self.min_val and self.max_val
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
        
        # Register buffers for non-learnable parameters
        self.register_buffer('data_min', None)
        self.register_buffer('data_range', None)

    def fit(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=self.dtype, device=self.device)
        
        # Safely handle NaN/Inf before calculation
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
        
        # Use registered buffers for scaling
        return (x - self.data_min) / self.data_range

    def denormalize(self, x):
        if self.data_min is None or self.data_range is None:
            raise RuntimeError("ScalingTransform has not been fitted yet.")
        
        return x * self.data_range + self.data_min

    def to(self, *args, **kwargs):
        # The super().to() call handles moving the registered buffers.
        super().to(*args, **kwargs)
        
        # Update device and dtype attributes for consistency
        if 'device' in kwargs:
            self.device = kwargs['device']
        elif len(args) > 0 and isinstance(args[0], (str, torch.device)):
            self.device = args[0]
            
        if 'dtype' in kwargs:
            self.dtype = kwargs['dtype']
        elif len(args) > 0 and isinstance(args[0], torch.dtype):
            self.dtype = args[0]

        return self

class ZTransform(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.device = kwargs.get('device', 'cpu')
        self.dtype = kwargs.get('dtype', torch.float32)
        # We don't want these to be part of the state dict at initialization
        self.register_buffer('mu', None)
        self.register_buffer('sigma', None)

    def fit(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=self.dtype, device=self.device)
        self.mu = torch.nanmean(data).detach()
        self.sigma = torch.std(torch.nan_to_num(data, 0, 0, 0)).detach()
        # Handle the edge case of zero standard deviation
        if self.sigma == 0:
            self.sigma = torch.tensor(1.0, dtype=self.dtype, device=self.device).detach()

    def forward(self, x):
        if self.mu is None or self.sigma is None:
            raise RuntimeError("ZTransform has not been fitted yet.")
        return (x - self.mu) / self.sigma

    def to(self, *args, **kwargs):
        # The super().to() call handles moving the registered buffers (mu, sigma)
        super().to(*args, **kwargs)
        # Update internal attributes for consistency if needed
        # args[0] is typically the device or dtype
        if args and isinstance(args[0], str):
            self.device = args[0]
        if args and isinstance(args[0], torch.dtype):
            self.dtype = args[0]
        
        # kwargs can also specify device and dtype
        if 'device' in kwargs:
            self.device = kwargs['device']
        if 'dtype' in kwargs:
            self.dtype = kwargs['dtype']

        return self
  