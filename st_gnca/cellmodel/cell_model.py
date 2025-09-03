import torch
from torch import nn

from xlstm import xLSTMBlockStack
from torch_geometric.nn import GATConv


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class xLSTMForecast(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, graph, cfg,
                 dropout=0.15, device=DEVICE, dtype=torch.float32, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        self.dtype = dtype
        self.graph = graph
        self.X_batch_graph = kwargs.get('X_batch_graph', None)

        # GAT Layer
        self.gat_layer = GATConv(
                    in_channels=(4+5),
                    out_channels=hidden_dim
                ).to(dtype=dtype)
        
        # XLSTM Block Stack
        self.xlstm = xLSTMBlockStack(cfg).to(dtype=dtype)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim).to(dtype=dtype)

        # Ensure all parameters are on correct device and dtype
        self.to(device=device, dtype=dtype)

    def _prepare_xlstm_input(self, x):
        # x shape: [batch_size, seq_len, feature_dim]
        # We need to ensure the input to xLSTM is of shape [batch_size, seq_len, hidden_dim]
        if x.size(2) != self.hidden_dim:
            x = nn.Linear(x.size(2), self.hidden_dim).to(device=self.device, dtype=self.dtype)(x)
        return x

    def forward(self, x):
        print(f"Input x shape: {x.shape}") # Input x shape: torch.Size([32, 10, 9])
        sequence_out = []

        if self.X_batch_graph is not None:
            for t in range(x.size(1)):
                #extract time step t
                xt = self.X_batch_graph[:, t, :]
                print(f"Time step {t}, xt shape before GAT: {xt.shape}") #torch.Size([32, 9])

                # Apply GAT layer
                gat_out = self.gat_layer(xt, self.graph.edge_index)

                sequence_out.append(gat_out.unsqueeze(1))

            sequence_out = torch.stack(sequence_out, dim=1)
            print(f"Sequence out shape after GAT: {sequence_out.shape}")

        # Filtrar
        xlstm_input = self._prepare_xlstm_input(sequence_out)

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