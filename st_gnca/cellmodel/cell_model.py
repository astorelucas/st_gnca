import torch
from torch import nn

from xlstm import xLSTMBlockStack
from torch_geometric.nn import GATConv


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class xLSTMForecast(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, edge_index, cfg,
                 dropout=0.15, device=DEVICE, dtype=torch.float32, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        self.dtype = dtype
        self.edge_index = edge_index
        self.X_batch_graph = kwargs.get('X_batch_graph', None)

        # GAT Layer
        self.gat_layer = GATConv(
                    in_channels=1,
                    out_channels=hidden_dim
                ).to(dtype=dtype)
        
        # XLSTM Block Stack
        self.xlstm = xLSTMBlockStack(cfg).to(dtype=dtype)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim).to(dtype=dtype)

        # Ensure all parameters are on correct device and dtype
        self.to(device=device, dtype=dtype)

   #FALTA FAZER ESSA FUNCAO : 
    def _prepare_xlstm_input(self, x, sensor, gat_out):
        # Prepare input for xLSTM by concatenating GAT output with target sensor data
        for neighbor in sensor.neighbors:
            gat_out = torch.cat((gat_out, x[neighbor]), dim=-1)
        return gat_out

    def forward(self, x, sensor):
        print(f"Input x shape: {x.shape}") # Input x shape: torch.Size([32, 10, 9])
        sequence_out = []
        selected_indices = [4, 5, 6, 7, 8]  # Assuming first 4 indices are temporal features

        if self.X_batch_graph is not None:
            xt_filtered = self.X_batch_graph[:, :, selected_indices]
            print(f"Filtered x shape : {xt_filtered.shape}")
            for t in range(x.size(1)):
                #extract time step t
                xt = xt_filtered[:, t, :]
                print(f"Time step {t}, xt shape before GAT: {xt.shape}") #torch.Size([32, 9])

                # Apply GAT layer
                xt = xt.unsqueeze(-1)
                xt_flattened = xt.contiguous().view(-1, 1)
                gat_out = self.gat_layer(xt_flattened, self.edge_index)
                gat_out = gat_out.view(xt.size(0), -1, self.gat_layer.out_channels)

                sequence_out.append(gat_out)

            sequence_out = torch.stack(sequence_out, dim=1)
            print(f"Sequence out shape after GAT: {sequence_out.shape}")

        # Filtrar
        xlstm_input = self._prepare_xlstm_input(x, sensor, sequence_out)

        print(f"xLSTM input shape: {xlstm_input.shape}") #[batch, seq_len, hidden_dim]
        xlstm_out, _ = self.xlstm(xlstm_input)

        prediction = self.output_proj(xlstm_out[:, -1, :])

        print(f"Prediction shape: {prediction.shape}")
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