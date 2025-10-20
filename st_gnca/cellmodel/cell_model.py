import torch
from torch import nn

from xlstm import xLSTMBlockStack
from st_gnca.tokenizer.tokenizer import NeighborhoodTokenizer
from typing import Any


DEVICE = (
    torch.device('cuda') if torch.cuda.is_available()
    else torch.device('mps') if torch.backends.mps.is_available()
    else torch.device('cpu')
)

class xLSTMForecast(nn.Module):
    def __init__(self, feature_dim, output_dim, hidden_dim, edge_index, cfg,
                 dropout=0.2, device=DEVICE, dtype=torch.float32, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        self.dtype = dtype
        self.edge_index = edge_index
        self.graph = kwargs.get('graph', None)
        self.max_graph_degree = max(dict(self.graph.degree()).values())
        self.temp_dim = kwargs.get('temp_dim', 4)  # Default temporal embedding dimension
        
        self.tokenizer = NeighborhoodTokenizer(
            graph=self.graph,
            edge_index=self.edge_index,
            temp_dim=self.temp_dim,
            hidden_dim=self.hidden_dim,
            dtype=self.dtype
        ).to(device=device, dtype=dtype)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        self.input_mapper = nn.Linear(feature_dim, hidden_dim).to(dtype=dtype)
        
        # XLSTM Block Stack torch.Size([32, 12, 64])
        self.xlstm = xLSTMBlockStack(cfg).to(dtype=dtype)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim).to(dtype=dtype)

        # Ensure all parameters are on correct device and dtype
        self.to(device=device, dtype=dtype)


    def forward(self, sensor, gat_emb, time_emb, subset_nodes):
        

        tokenized_data = self.tokenizer.forward(gat_emb, time_emb, sensor, subset_nodes)

        mapped_x = self.input_mapper(tokenized_data)

        xlstm_in = self.dropout(mapped_x)

        xlstm_out = self.xlstm(xlstm_in)

        xlstm_output = self.dropout(xlstm_out)

        prediction = self.output_proj(xlstm_output[:, -1, :])

        return prediction
    
    def to(self, *args, **kwargs):

      self = super().to(*args, **kwargs)
      for arg in args:
          if isinstance(arg, torch.dtype):
              self.dtype = arg
          elif isinstance(arg, (str, torch.device)):
              self.device = arg if isinstance(arg, str) else arg.type
      return self
    
class LSTMForecast(nn.Module):
    def __init__(self, feature_dim: int, output_dim: int, hidden_dim: int, edge_index: Any, 
                 dropout: float = 0.2, device: torch.device = None, dtype: torch.dtype = torch.float32, **kwargs):
        
        super().__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if torch.backends.mps.is_available():
                device = torch.device('mps')
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        self.dtype = dtype
        self.edge_index = edge_index
        self.graph = kwargs.get('graph', None)
        self.max_graph_degree = max(dict(self.graph.degree()).values())
        self.temp_dim = kwargs.get('temp_dim', 4)
        self.num_layers = kwargs.get('num_layers', 4)

        self.tokenizer = NeighborhoodTokenizer(
            graph=self.graph,
            edge_index=self.edge_index,
            temp_dim=self.temp_dim,
            hidden_dim=self.hidden_dim,
            dtype=self.dtype
        ).to(device=self.device, dtype=self.dtype)

        self.dropout = nn.Dropout(p=dropout)

        self.input_mapper = nn.Linear(self.feature_dim, self.hidden_dim, dtype=self.dtype).to(self.device)

        self.input_norm = nn.LayerNorm(self.hidden_dim, dtype=self.dtype).to(self.device)

        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,  
            batch_first=True,
            dropout=dropout, 
            dtype=self.dtype
        ).to(self.device)

        self.output_proj = nn.Linear(self.hidden_dim, self.output_dim, dtype=self.dtype).to(self.device)

        self.to(device=self.device, dtype=self.dtype)


    def forward(self, sensor: torch.Tensor, gat_emb: torch.Tensor, time_emb: torch.Tensor, subset_nodes: torch.Tensor):
        
        tokenized_data = self.tokenizer.forward(gat_emb, time_emb, sensor, subset_nodes)
        
        mapped_x = self.input_mapper(tokenized_data)

        mapped_x = self.input_norm(mapped_x)

        lstm_in = self.dropout(mapped_x)
        
        lstm_out, _ = self.lstm(lstm_in)

        lstm_output = self.dropout(lstm_out)

        prediction = self.output_proj(lstm_output[:, -1, :])
        
        return prediction
    
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        if 'device' in kwargs:
            self.device = kwargs['device']
        if 'dtype' in kwargs:
            self.dtype = kwargs['dtype']
        return self
