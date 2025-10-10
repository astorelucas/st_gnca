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
    def __init__(self, feature_dim, output_dim, hidden_dim, edge_index, graph, device=None, **kwargs):
        super(TFTForecast, self).__init__()

        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.edge_index = edge_index
        self.graph = graph
        self.device = device if device is not None else torch.device('cpu')

        # TFT hyperparameters - good defaults based on research
        self.max_encoder_length = 12  # matches sequence_len from original
        self.max_prediction_length = output_dim

        # Create a simple input transformation layer
        # TFT expects specific input format, so we'll create an adapter
        # self.input_projection = nn.Linear(feature_dim, hidden_dim)
        self.input_projection = None # replaced it so it runs dynamicly in the foward phase

        # Output projection to match expected dimensions
        self.output_projection = nn.Linear(hidden_dim, output_dim)

        # TFT core parameters - optimized defaults
        tft_params = {
            'hidden_size': hidden_dim,           # 64
            'attention_head_size': 8,            # Good default for traffic data
            'dropout': 0.3,                      # Moderate dropout
            'hidden_continuous_size': hidden_dim // 2,  # 32
            'output_size': output_dim,           # 3 (horizon)
            'loss': QuantileLoss(),
            'learning_rate': 0.001,              # Standard learning rate
            'reduce_on_plateau_patience': 4,
        }

        # Store TFT parameters for later use
        self.tft_params = tft_params

        # We'll create a simplified TFT-like architecture for this specific use case
        # Since the original TFT requires TimeSeriesDataSet, we'll build core components

        # LSTM encoder-decoder (core of TFT)
        self.encoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=4,
            dropout=0.2,
            batch_first=True
        )

        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=4,
            dropout=0.2,
            batch_first=True
        )

        # Multi-head attention (key component of TFT)
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.2,
            batch_first=True
        )

        # Gated residual networks (GRN) - simplified version
        self.grn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(0.2)
        )

        self.grn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(0.2)
        )

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.layer_norm3 = nn.LayerNorm(hidden_dim)

        self.to(self.device)

    def forward(self, *args, **kwargs):
        """
        Forward pass through TFT-inspired architecture.

        Flexible signature to match original xLSTMForecast interface.
        The original likely had signature like: forward(self, x, edge_index, temporal_features, spatial_features)
        or similar variations.
        """


        # Handle different argument patterns
        if len(args) == 1:
            # Simple case: forward(x)
            x = args[0]
        elif len(args) >= 2:
            # Complex case: forward(x, edge_index, ...) or forward(x, temporal_features, ...)
            x = args[2]
    
            # Additional arguments can be edge_index, temporal_features, etc.
            # For now, we'll use just the first argument (x) and ignore the rest
            # You can modify this based on your specific needs
        else:
            raise ValueError(f"Expected at least 1 argument, got {len(args)}")


        # Ensure x has the right shape: [batch_size, sequence_length, feature_dim]
        if len(x.shape) == 2:
            # If x is [batch_size, feature_dim], add sequence dimension
            x = x.unsqueeze(1)  # [batch_size, 1, feature_dim]
        elif len(x.shape) == 3:
            # Already in correct format [batch_size, seq_len, feature_dim]
            pass
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        if x.device != self.device:
            x = x.to(self.device)

        #print(f"Actual input shape: {x.shape}")  # ADD THIS LINE
        #print(f"Expected feature_dim: {self.feature_dim}")  # ADD THIS LINE

        # input size detection and projection
        batch_size, seq_len, actual_input_dim = x.shape

        # If input projection doesn't match, recreate it dynamically
        if not hasattr(self, '_input_dim_set') or self.input_projection.in_features != actual_input_dim:
            self.input_projection = nn.Linear(actual_input_dim, self.hidden_dim).to(self.device)
            self._input_dim_set = True
            print(f"Adjusted input projection for actual input dim: {actual_input_dim}")

        # Input projection
        x = self.input_projection(x)  # [batch_size, seq_len, hidden_dim]

        # LSTM encoder
        encoded, (h_enc, c_enc) = self.encoder_lstm(x)
        encoded = self.layer_norm1(encoded)

        # Apply GRN 1
        grn1_out = self.grn1(encoded)
        encoded = self.layer_norm2(encoded + grn1_out)  # Residual connection

        # Multi-head attention
        attn_out, _ = self.multihead_attention(encoded, encoded, encoded)
        encoded = self.layer_norm3(encoded + attn_out)  # Residual connection

        # Apply GRN 2
        grn2_out = self.grn2(encoded)
        encoded = encoded + grn2_out  # Residual connection

        # For prediction, we can either use the last time step or decode multiple steps
        if seq_len >= self.max_prediction_length:
            # Use the last few time steps for prediction
            decoder_input = encoded[:, -self.max_prediction_length:, :]
        else:
            # If sequence is shorter than prediction length, use all of it
            decoder_input = encoded

        # LSTM decoder for final prediction
        decoded, _ = self.decoder_lstm(decoder_input, (h_enc, c_enc))

        # Output projection
        output = self.output_projection(decoded)  # [batch_size, seq_steps, output_dim]

        # Take mean across sequence dimension to get final prediction
        # This matches the expected output format for single-step prediction
        if output.shape[1] > 1:
            output = output.mean(dim=1)  # [batch_size, output_dim]
        else:
            output = output.squeeze(1)  # [batch_size, output_dim]

        return output

