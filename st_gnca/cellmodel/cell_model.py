import torch
from torch import nn

from st_gnca.modules.transformers import Transformer, get_config as transformer_get_config

from st_gnca.modules.moe import SparseMixtureOfExperts
from st_gnca.common import activations, dtypes, get_device
from st_gnca.datasets.PEMS import get_config as pems_get_config

from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig, sLSTMBlockConfig, mLSTMBlockConfig, sLSTMLayerConfig, mLSTMLayerConfig, FeedForwardConfig


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_config(config):
  return CellModel(num_tokens = config.pop('num_tokens',10), 
                   dim_token = config.pop('dim_token',10),
                   num_transformers = config.pop('num_transformers',10), 
                   num_heads = config.pop('num_heads',10), 
                   transformer_feed_forward = config.pop('transformer_feed_forward',10), 
                   transformer_activation = config.pop('transformer_activation',10),
                   feed_forward = config.pop('feed_forward',1),
                   feed_forward_dim = config.pop('feed_forward_dim',100), 
                   feed_forward_activation = config.pop('feed_forward_activation',nn.GELU()),
                   device = config.pop('device',None), 
                   dtype = config.pop('dtype',torch.float32), 
                   **config)

def get_config(model, **extra):
  a = transformer_get_config(model.transformers[0])
  b = { 
    'num_tokens': model.num_tokens, 
    'dim_token': model.dim_token,
    'num_transformers': model.num_transformers,
    'feed_forward': model.feed_forward,
    'feed_forward_dim': model.feed_forward_dim, 
    'feed_forward_activation': model.activation,
    'device': model.device, 
    'dtype': model.dtype
  }
  if model.use_moe:
    b['use_moe'] = model.use_moe
    b['num_experts'] = model.moe.num_experts
  a |= b 
  a |= extra
  return a

def save_as(from_file, to_file, pems, **kwargs):
    DEVICE = pems.device
    DTYPE = pems.dtype
    if kwargs.get('config',None):
      model = load_config(kwargs.get('config',None))
    else:
      NTRANSF = kwargs.get('NTRANSF',10)
      NHEADS = kwargs.get('NHEADS',10)
      NTRANSFF = kwargs.get('NTRANSFF',10)
      TRANSFACT = kwargs.get('TRANSFACT',nn.GELU())
      MLP = kwargs.get('MLP',10)
      MLPD = kwargs.get('MLPD',10)
      MLPACT = kwargs.get('MLPACT',nn.GELU())      
      model = CellModel(num_tokens = pems.max_length, dim_token= pems.token_dim,
                num_transformers = NTRANSF, num_heads = NHEADS, transformer_feed_forward= NTRANSFF, 
                transformer_activation = TRANSFACT,
                feed_forward = MLP, feed_forward_dim = MLPD, feed_forward_activation = MLPACT,
                device = DEVICE, dtype = DTYPE)
      
    model.load_state_dict(torch.load(from_file, 
                                 weights_only=True,
                                 map_location=torch.device(DEVICE)), strict=False)
    extra = {}
    extra |= pems_get_config(pems)
    extra |= kwargs
    torch.save({
        'config': get_config(model, **extra),
        "weights": model.state_dict() }, 
        to_file)
    
def setup(file, pems, DEVICE):
    saved_config = torch.load(file)
    print(saved_config['config'])
    tmp = load_config(saved_config['config']).to(DEVICE)
    tmp.load_state_dict(saved_config['weights'], strict=False)
    pems = pems.to(DEVICE)
    pems.steps_ahead = saved_config['config']['steps_ahead']
    return tmp, pems

class CellModel(nn.Module):
  def __init__(self, num_tokens, dim_token,
               num_transformers, num_heads, transformer_feed_forward, transformer_activation = nn.GELU(),
               feed_forward = 1, feed_forward_dim = 100, feed_forward_activation = nn.ReLU(),
               device = DEVICE, dtype = torch.float64, **kwargs):
    super().__init__()
    self.num_tokens = num_tokens
    self.num_transformers = num_transformers
    self.dim_token = dim_token
    self.device = device
    self.dtype = dtype
    self.feed_forward = feed_forward
    self.feed_forward_dim = feed_forward_dim

    self.use_moe = kwargs.get('use_moe',False)
    

    self.transformers = nn.ModuleList([Transformer(num_heads, self.num_tokens, dim_token, transformer_feed_forward, transformer_activation,
                         dtype=self.dtype, device=self.device, **kwargs)
                         for k in range(num_transformers)])

    self.flat = nn.Flatten(1)

    if not self.use_moe:

      self.linear = nn.ModuleList()
      for l in range(feed_forward):
        in_dim = self.num_tokens * self.dim_token if l == 0 else feed_forward_dim
        out_dim = 1 if l == feed_forward-1 else feed_forward_dim
        self.linear.append(nn.Linear(in_dim, out_dim, dtype=self.dtype, device=self.device))

    else:

      num_experts = kwargs.get('num_experts',4)

      self.moe = SparseMixtureOfExperts(dtype=self.dtype, device=self.device,
                                        num_experts = num_experts, activate = 1, 
                                        input_dim = self.num_tokens * self.dim_token,
                                        router_input_dim = self.dim_token,
                                        output_dim = feed_forward_dim,
                                        expert_hidden_dim = feed_forward_dim,
                                        num_layers = self.feed_forward,
                                        activation = feed_forward_activation,
                                        use_moe = False,
                                        #router_hidden_dim = 8,
                                        router_type='lsh')
      self.final = nn.Linear(feed_forward_dim, 1, dtype=self.dtype, device=self.device)

    self.activation = feed_forward_activation
    self.drop = nn.Dropout(.15)


  def forward(self, x):
    for transformer in self.transformers:
      x = transformer(x)
    z = self.flat(x)
    if not self.use_moe:
      for linear in self.linear:
        z = self.activation(linear(self.drop(z)))
    else:
      z = self.moe(z)
      z = self.activation(self.final(self.drop(z)))
    return z

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]
    for k in range(self.num_transformers):
      self.transformers[k] = self.transformers[k].to(*args, **kwargs)
    if not self.use_moe:
      for k in range(self.feed_forward):
        self.linear[k] = self.linear[k].to(*args, **kwargs)
    else:
      self.moe = self.moe.to(*args, **kwargs)
      self.final = self.final.to(*args, **kwargs)
    return self

  def train(self, *args, **kwargs):
    super().train(*args, **kwargs)
    for k in range(self.num_transformers):
      self.transformers[k] = self.transformers[k].train(*args, **kwargs)
    if not self.use_moe:
      for k in range(self.feed_forward):
        self.linear[k] = self.linear[k].train(*args, **kwargs)
    else:
      self.moe = self.moe.train(*args, **kwargs)
      self.final = self.final.train(*args, **kwargs)
    return self
  

  # LSTM-based CellModel implementation

class CellModel_LSTM(nn.Module):
    def __init__(self, num_tokens, dim_token, hidden_size=128, num_layers=2, dropout=0.15, device=DEVICE, dtype=torch.float32, **kwargs):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim_token = dim_token
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype

        # LSTM expects input shape (batch, seq, feature)
        self.lstm = nn.LSTM(
            input_size=dim_token,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        ).to(device=device, dtype=dtype)

        self.fc = nn.Linear(hidden_size, 1, dtype=dtype, device=device)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len=num_tokens, dim_token)
        # LSTM returns output for all timesteps, we use the last one
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take last time step
        out = self.drop(out)
        out = self.fc(out)
        return out
    
    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.lstm = self.lstm.train(*args, **kwargs)
        self.fc = self.fc.train(*args, **kwargs)
        self.drop = self.drop.train(*args, **kwargs)
        return self
    
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        if isinstance(args[0], str):
            self.device = args[0]
        else:
            self.dtype = args[0]
        self.lstm = self.lstm.to(*args, **kwargs)
        self.fc = self.fc.to(*args, **kwargs)
        self.drop = self.drop.to(*args, **kwargs)
        return self
    

class CellModel_xLSTM(nn.Module):
    def __init__(self, num_nodes, input_len, output_len, hidden_dim, cfg, dropout=0.15, device=DEVICE, dtype=torch.float64, **kwargs):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_dim = hidden_dim
        self.device = device
        self.dtype = dtype
        self.token_dim = 9
        # Projects each timestep's node vector (length = num_nodes) to hidden_dim
        self.input_proj = nn.Linear(self.token_dim, self.hidden_dim)

        # The xLSTM block takes input of shape [B, T, hidden_dim]
        self.xlstm = xLSTMBlockStack(cfg)

        # Output projection: from hidden_dim → (num_nodes * output_len)
        self.output_proj = nn.Linear(hidden_dim, num_nodes * output_len)

        # Add dropout layer
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, N, T, D]
        B, N, T, D = x.shape
        x = x.view(B * N, T, D)
        x = self.input_proj(x)
        x = self.xlstm(x)
        x = x[:, -1, :]
        x = self.output_proj(x)
        x = x.view(B, N, self.output_len, self.num_nodes)
        x = x.mean(dim=1)  # Aggregate over nodes if needed
        return x  # [B, output_len, num_nodes]
    
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.xlstm = self.xlstm.to(*args, **kwargs)
        self.drop = self.drop.to(*args, **kwargs)
        return self

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.xlstm = self.xlstm.train(*args, **kwargs)
        self.drop = self.drop.train(*args, **kwargs)
        return self
    
class xLSTMForecast2(nn.Module):
    def __init__(self, token_dim, max_length, output_len, hidden_dim, cfg, dropout=0.15, device="cuda", dtype=torch.float64):
        super().__init__()
        self.token_dim = token_dim    # Embedding size of each token (spatial + value + temporal)
        self.max_length = max_length  # Max nodes (main + neighbors)
        self.output_len = output_len
        self.hidden_dim = hidden_dim
        self.device = device
        self.dtype = dtype

        # Project token_dim to hidden_dim
        self.input_proj = nn.Linear(token_dim, hidden_dim)
        
        # xLSTM (from NX-AI)
        self.xlstm = xLSTMBlockStack(cfg)
        
        # Predict output_len steps per node
        self.output_proj = nn.Linear(hidden_dim, max_length * output_len)

    def forward(self, x):
        # x shape: (B, max_length, token_dim)
        B = x.shape[0]
        
        # Project tokens to hidden_dim
        x = self.input_proj(x)  # -> (B, max_length, H)
        
        # Process with xLSTM
        x = self.xlstm(x)       # -> (B, max_length, H)
        
        # Use last timestep (or mean/max pooling)
        x = x[:, -1, :]         # -> (B, H)
        
        # Predict output_len steps per node
        x = self.output_proj(x)  # -> (B, max_length * output_len)
        
        # Reshape to (B, output_len, max_length)
        return x.view(B, self.output_len, self.max_length)
    
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

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

        # Graph Attention Network (GAT)
        self.gat = GATConv(
            in_channels=9,  # token_dim
            out_channels=32,  # 128/4 heads = 32
            heads=4,  # 4 × 32 = 128
            concat=True,
            dropout=0.15,
            edge_dim=1
        )

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
        
        # Process through GAT - NOW PROPERLY DEFINED
        x_gat = self.gat(x_flat, edge_index, edge_attr)
        
        # Reshape back with dimension validation
        try:
            x_gat = x_gat.reshape(B, T, -1)
            if x_gat.shape[-1] != self.hidden_dim:
                raise ValueError(f"GAT output dim {x_gat.shape[-1]} != hidden_dim {self.hidden_dim}")
        except RuntimeError as e:
            raise RuntimeError(
                f"Reshape failed from {x_gat.shape} to [{B}, {T}, -1]. "
                f"GAT out_channels: {self.gat.out_channels}, heads: {self.gat.heads}"
            ) from e
        
        # Continue with XLSTM
        x_lstm = self.xlstm(x_gat)
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