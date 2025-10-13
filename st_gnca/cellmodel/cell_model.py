import torch
from torch import nn
import numpy as np

from xlstm import xLSTMBlockStack
from torch_geometric.nn import GATConv
from st_gnca.tokenizer.tokenizer import NeighborhoodTokenizer
from typing import Dict, Any
from pytorch_forecasting.metrics import QuantileLoss

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
        # print(f"xlstm_output {xlstm_output.shape}") # torch.Size([32, 12, 32])
        # print(f"xlstm_output {xlstm_output.dtype}")
        # The prediction is based on the final time step's output
        # xlstm_output[:, -1, :] gets the hidden state for the last time step

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

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Input projection layer
        # The input to this mapper is the tokenized data
        self.input_mapper = nn.Linear(self.feature_dim, self.hidden_dim, dtype=self.dtype).to(self.device)

        # LSTM layer
        # batch_first=True means input/output tensors are [batch, seq, features]
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,  # You can increase this for deeper models
            batch_first=True,
            dropout=dropout, # Dropout is applied to the output of each LSTM layer except the last one.
            dtype=self.dtype
        ).to(self.device)

        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, self.output_dim, dtype=self.dtype).to(self.device)

        # Ensure all parameters are on correct device and dtype
        self.to(device=self.device, dtype=self.dtype)


    def forward(self, sensor: torch.Tensor, gat_emb: torch.Tensor, time_emb: torch.Tensor, subset_nodes: torch.Tensor):
        
        # Tokenize the input data using the neighborhood tokenizer
        tokenized_data = self.tokenizer.forward(gat_emb, time_emb, sensor, subset_nodes)
        
        # Apply the input mapper to project features to hidden dimension
        mapped_x = self.input_mapper(tokenized_data)

        # Apply dropout to the input
        lstm_in = self.dropout(mapped_x)
        
        # Pass the data through the LSTM layer
        # The LSTM returns the full sequence output and the final hidden state
        lstm_out, _ = self.lstm(lstm_in)

        # Apply dropout to the LSTM output before the final projection
        lstm_output = self.dropout(lstm_out)

        # The prediction is based on the final time step's output
        # lstm_output[:, -1, :] gets the hidden state for the last time step
        prediction = self.output_proj(lstm_output[:, -1, :])
        
        return prediction
    
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        if 'device' in kwargs:
            self.device = kwargs['device']
        if 'dtype' in kwargs:
            self.dtype = kwargs['dtype']
        return self

# Custom TFT-based cell model to replace xLSTMForecast
class TFTForecast(nn.Module):
    """
    TFT-based cell model that replaces xLSTMForecast.
    Adapts TFT for graph neural cellular automata framework.
    """
    def __init__(self, feature_dim, output_dim, hidden_dim, edge_index, cfg=None, 
                 dropout=0.2, device=DEVICE, dtype=torch.float32, **kwargs):
        super().__init__()

        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.dtype = dtype
        self.edge_index = edge_index
        self.graph = kwargs.get('graph', None)
        self.max_graph_degree = max(dict(self.graph.degree()).values())
        self.temp_dim = kwargs.get('temp_dim', 4)  # Default temporal embedding dimension
        
        # Initialize tokenizer - this was missing!
        self.tokenizer = NeighborhoodTokenizer(
            graph=self.graph,
            edge_index=self.edge_index,
            temp_dim=self.temp_dim,
            hidden_dim=self.hidden_dim,
            dtype=self.dtype
        ).to(device=device, dtype=dtype)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Input projection to map tokenized features to hidden dimension
        self.input_mapper = nn.Linear(feature_dim, hidden_dim).to(dtype=dtype)

        # LSTM encoder-decoder (core of TFT)
        self.encoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=4,
            dropout=dropout,
            batch_first=True
        ).to(dtype=dtype)

        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=4,
            dropout=dropout,
            batch_first=True
        ).to(dtype=dtype)

        # Multi-head attention (key component of TFT)
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        ).to(dtype=dtype)

        # Gated residual networks (GRN) - simplified version
        self.grn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        ).to(dtype=dtype)

        self.grn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        ).to(dtype=dtype)

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim).to(dtype=dtype)
        self.layer_norm2 = nn.LayerNorm(hidden_dim).to(dtype=dtype)
        self.layer_norm3 = nn.LayerNorm(hidden_dim).to(dtype=dtype)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim).to(dtype=dtype)

        # Ensure all parameters are on correct device and dtype
        self.to(device=device, dtype=dtype)

    def forward(self, sensor, gat_emb, time_emb, subset_nodes):
        """
        Forward pass through TFT-inspired architecture.
        
        Args:
            sensor: Sensor data tensor
            gat_emb: Graph attention embeddings
            time_emb: Temporal embeddings
            subset_nodes: Subset of nodes to process
            
        Returns:
            prediction: Output predictions [batch_size, output_dim]
        """
        
        # Tokenize the input data using the neighborhood tokenizer
        # This matches the pattern from xLSTMForecast and LSTMForecast
        tokenized_data = self.tokenizer.forward(gat_emb, time_emb, sensor, subset_nodes)
        # Expected shape: [batch_size, seq_len, feature_dim]
        
        # Apply the input mapper to project features to hidden dimension
        mapped_x = self.input_mapper(tokenized_data)
        # Shape: [batch_size, seq_len, hidden_dim]
        
        # Apply dropout to the input
        x = self.dropout(mapped_x)
        
        # LSTM encoder
        encoded, (h_enc, c_enc) = self.encoder_lstm(x)
        encoded = self.layer_norm1(encoded)

        # Apply GRN 1 with residual connection
        grn1_out = self.grn1(encoded)
        encoded = self.layer_norm2(encoded + grn1_out)

        # Multi-head attention with residual connection
        attn_out, _ = self.multihead_attention(encoded, encoded, encoded)
        encoded = self.layer_norm3(encoded + attn_out)

        # Apply GRN 2 with residual connection
        grn2_out = self.grn2(encoded)
        encoded = encoded + grn2_out

        # LSTM decoder for final prediction
        decoded, _ = self.decoder_lstm(encoded, (h_enc, c_enc))
        
        # Apply dropout to the decoder output before the final projection
        decoded_output = self.dropout(decoded)

        # The prediction is based on the final time step's output
        # This matches the pattern from xLSTMForecast and LSTMForecast
        prediction = self.output_proj(decoded_output[:, -1, :])
        # Shape: [batch_size, output_dim]

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
