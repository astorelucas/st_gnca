import torch
from torch import nn
import numpy as np

from xlstm import xLSTMBlockStack
from torch_geometric.nn import GATConv
from st_gnca.tokenizer.tokenizer import NeighborhoodTokenizer


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


    def forward(self, x, sensor, spatial_embedder, x_time):
        

        tokenized_data = self.tokenizer.forward(x, spatial_embedder, x_time, sensor)
        # print(f"Tokenized data shape: {tokenized_data.shape}")  # Example output: torch.Size([32, 12, 134])
        # print(f"Tokenized data: {tokenized_data[0, 0, :]}")  # Example output: tensor([...])

        # print(f"xLSTM input shape before mapping: {tokenized_data.shape}") #torch.Size([32, 12, 134])
        mapped_x = self.input_mapper(tokenized_data)
        # print(f"xLSTM input shape: {mapped_x.shape}") #torch.Size([32, 12, 64])

        # Apply dropout to the input
        xlstm_in = self.dropout(mapped_x)
        # print(f"xLSTM input shape after dropout: {xlstm_in.shape}")

        xlstm_out = self.xlstm(xlstm_in)
        # print(f"xLSTM output shape/: {xlstm_out.shape}") #torch.Size([32, 12, 64])

        # Apply dropout to the XLSTM output before the final projection
        xlstm_output = self.dropout(xlstm_out)

        prediction = self.output_proj(xlstm_output[:, -1, :])
        # print(f"Prediction shape: {prediction.shape}") #torch.Size([32, 3]) 

        # print(f"Prediction shape: {prediction.shape}")
        return prediction
    
    def to(self, *args, **kwargs):

      self = super().to(*args, **kwargs)
      # Update device/dtype attributes if specified

      for arg in args:
          if isinstance(arg, torch.dtype):
              self.dtype = arg
          elif isinstance(arg, (str, torch.device)):
              self.device = arg if isinstance(arg, str) else arg.type
      return self