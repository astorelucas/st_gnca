import torch
from torch import nn

from xlstm import xLSTMBlockStack
from torch_geometric.nn import GATConv


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class xLSTMForecast(nn.Module):
    def __init__(self, input_dim, max_length, output_dim, hidden_dim, cfg, 
                 dropout=0.15, device="cuda", dtype=torch.float32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        self.dtype = dtype

        # GAT Layer
        self.gat_layer = GATConv(
                    in_channels=input_dim,
                    out_channels=hidden_dim
                ).to(dtype=dtype)
        
        # XLSTM Block Stack
        self.xlstm = xLSTMBlockStack(cfg).to(dtype=dtype)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim).to(dtype=dtype)

        # Ensure all parameters are on correct device and dtype
        self.to(device=device, dtype=dtype)

    def forward(self, x, graph):

        sequence_out = []

        for t in range(x.size(1)):
            xt = x[:, t, :]

            # Apply GAT layer
            gat_out = self.gat_layer(xt, graph)

            sequence_out.append(gat_out.unsqueeze(1))


        sequence_out = torch.stack(sequence_out, dim=1)

        xlstm_input = sequence_out.view(sequence_out.size(0), sequence_out.size(1), -1)

        xlstm_out, _ = self.xlstm(xlstm_input)

        prediction = self.output_proj(xlstm_out[:, -1, :])

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