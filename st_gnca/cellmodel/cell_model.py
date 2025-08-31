import torch
from torch import nn

from st_gnca.modules.transformers import Transformer, get_config as transformer_get_config

from st_gnca.modules.moe import SparseMixtureOfExperts
from st_gnca.common import activations, dtypes, get_device
from st_gnca.datasets.PEMS import get_config as pems_get_config

from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig, sLSTMBlockConfig, mLSTMBlockConfig, sLSTMLayerConfig, mLSTMLayerConfig, FeedForwardConfig


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class xLSTMForecast(nn.Module):
    def __init__(self, token_dim, max_length, output_len, hidden_dim, cfg, 
                 dropout=0.15, device="cuda", dtype=torch.float32):
        super().__init__()
        self.token_dim = token_dim
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.output_len = output_len
        self.device = device
        self.dtype = dtype

        # XLSTM Block Stack
        self.xlstm = xLSTMBlockStack(cfg).to(dtype=dtype)

        # Output projection
        self.output_proj = nn.Linear(
            hidden_dim, 
            output_len * max_length
        ).to(dtype=dtype)

        # Ensure all parameters are on correct device and dtype
        self.to(device=device, dtype=dtype)

    def forward(self, x, edge_index, edge_attr=None):
        # Validate input dtype
        if x.dtype != self.dtype:
            x = x.to(dtype=self.dtype)
        if edge_attr is not None and edge_attr.dtype != self.dtype:
            edge_attr = edge_attr.to(dtype=self.dtype)
        
        B, T, N = x.shape  # Batch, Timesteps, Node features

        # Flatten spatial-temporal dimensions for GAT
        x_flat = x.reshape(B * T, N)
        # Continue with XLSTM
        
        x_lstm = self.xlstm(x_flat)
        output = self.output_proj(x_lstm[:, -1, :])
        return output.reshape(B, self.output_len, -1)
    
    def to(self, *args, **kwargs):
      """Override to ensure consistent device/dtype handling"""
      self = super().to(*args, **kwargs)
      # Update device/dtype attributes if specified
      for arg in args:
          if isinstance(arg, torch.dtype):
              self.dtype = arg
          elif isinstance(arg, (str, torch.device)):
              self.device = arg if isinstance(arg, str) else arg.type
      return self