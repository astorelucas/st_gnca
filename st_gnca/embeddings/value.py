import torch
from torch import nn
import numpy as np

class ValueEmbedding(nn.Module):
    def __init__(self, data, method="ztransform", **kwargs):
        super().__init__()
        self.device = kwargs.get('device', 'cpu')
        self.dtype = kwargs.get('dtype', torch.float32)
        self.method = method

        data = np.array(data)

        if method == 'ztransform':
            self.embedder = ZTransform(data, **kwargs)
        elif method == 'minmaxn':
            self.embedder = ScalingTransform(data, **kwargs)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def forward(self, x):
        return self.embedder(x)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if isinstance(args[0], str):
            self.device = args[0]
        else:
            self.dtype = args[0]
        self.embedder = self.embedder.to(*args, **kwargs)
        return self
    
class ZTransform(nn.Module):
    def __init__(self, data, axis=0, eps=1e-8, **kwargs):
        super().__init__()
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)

        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.tensor(std + eps, dtype=torch.float32))

    def forward(self, x):
        return (x - self.mean) / self.std

class ScalingTransform(nn.Module):
    def __init__(self, data, axis=0, eps=1e-8, **kwargs):
        super().__init__()
        data_min = np.min(data, axis=axis, keepdims=True)
        data_max = np.max(data, axis=axis, keepdims=True)

        self.register_buffer('min', torch.tensor(data_min, dtype=torch.float32))
        self.register_buffer('range', torch.tensor(data_max - data_min + eps, dtype=torch.float32))

    def forward(self, x):
        return (x - self.min) / self.range
