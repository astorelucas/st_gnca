import torch
from torch import nn
import numpy as np

from xlstm import xLSTMBlockStack
from torch_geometric.nn import GATConv


DEVICE = (
    torch.device('cuda') if torch.cuda.is_available()
    else torch.device('mps') if torch.backends.mps.is_available()
    else torch.device('cpu')
)

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
        self.graph = kwargs.get('graph', None)
        self.max_graph_degree = max(dict(self.graph.degree()).values())
        self.temp_dim = kwargs.get('temp_dim', 4)  # Default temporal embedding dimension

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # # GAT Layer
        # self.gat_layer = GATConv(
        #             in_channels=1,
        #             out_channels=hidden_dim
        #         ).to(dtype=dtype)
        
        self.input_mapper = nn.Linear(input_dim, hidden_dim).to(dtype=dtype)
        
        # XLSTM Block Stack
        self.xlstm = xLSTMBlockStack(cfg).to(dtype=dtype)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim).to(dtype=dtype)

        # Ensure all parameters are on correct device and dtype
        self.to(device=device, dtype=dtype)

        src, dst = self.edge_index
        self.neighbors_list = [[] for _ in range(self.graph.number_of_nodes())]
        for s, d in zip(src.tolist(), dst.tolist()):
            self.neighbors_list[s].append(d)
            self.neighbors_list[d].append(s)

    def _tokenizer(
        self,
        raw_features: torch.Tensor,
        gat_features: torch.Tensor,
        temporal_features: torch.Tensor,
        target_sensor_idx: int
    ):
        """
        Finds a target sensor and its neighbors, then aggregates and pads raw, temporal,
        and GAT features into a single tensor for that neighborhood.
        """

        # --- 1. Identify the nodes in the neighborhood (all in torch, no numpy) ---
        neighbors = torch.tensor(self.neighbors_list[target_sensor_idx], device=self.edge_index.device)
        neighbors = torch.unique(neighbors)  # remove duplicates

        num_neighbors = neighbors.size(0)
        assert 0 < num_neighbors <= self.max_graph_degree, \
            f"Sensor {target_sensor_idx} has {num_neighbors} neighbors, which exceeds max graph degree {self.max_graph_degree} or is zero."
        
        # include the target itself at index 0
        all_indices = torch.cat([torch.tensor([target_sensor_idx], device=neighbors.device), neighbors])

        # --- 2. Filter input tensors for the neighborhood ---
        raw_f = raw_features[:, :, all_indices].unsqueeze(-1)               # (B, S, N, 1)
        gat_f = gat_features[:, :, all_indices, :]                          # (B, S, N, H)
        combined = torch.cat((raw_f, gat_f), dim=-1)                        # (B, S, N, 1+H)
        
        # --- 3. Pad with zeros if necessary ---
        pad_nodes = self.max_graph_degree - num_neighbors
        if pad_nodes > 0:
            pad = torch.zeros(
                (combined.size(0), combined.size(1), pad_nodes, combined.size(-1)),
                dtype=combined.dtype,
                device=combined.device
            )
            combined = torch.cat((combined, pad), dim=2)                    # (B, S, max_deg, 1+H)

        # --- 4. Flatten neighborhood features and concat with temporal ---
        flat = combined.flatten(start_dim=2)                                # (B, S, max_deg*(1+H))
        final = torch.cat((temporal_features, flat), dim=-1)                # (B, S, T + max_deg*(1+H)) 
        return final

    # def _tokenizer(
    #     self,
    #     raw_features: torch.Tensor,
    #     gat_features: torch.Tensor,
    #     temporal_features: torch.Tensor,
    #     target_sensor_idx: int
    # ):
    #     """
    #     Finds a target sensor and its neighbors, then aggregates and pads raw, temporal,
    #     and GAT features into a single tensor for that neighborhood.
    #     """

    #     # --- 1. Identify the nodes in the neighborhood (all in torch, no numpy) ---
    #     mask_src = self.edge_index[0] == target_sensor_idx
    #     mask_dst = self.edge_index[1] == target_sensor_idx
    #     neighbors = torch.cat([self.edge_index[1, mask_src], self.edge_index[0, mask_dst]])
    #     neighbors = torch.unique(neighbors)  # remove duplicates

    #     num_neighbors = neighbors.size(0)
    #     assert 0 < num_neighbors <= self.max_graph_degree, \
    #         f"Sensor {target_sensor_idx} has {num_neighbors} neighbors, which exceeds max graph degree {self.max_graph_degree} or is zero."

    #     # include the target itself at index 0
    #     all_indices = torch.cat([torch.tensor([target_sensor_idx], device=neighbors.device), neighbors])

    #     # --- 2. Filter input tensors for the neighborhood ---
    #     raw_f = raw_features[:, :, all_indices].unsqueeze(-1)               # (B, S, N, 1)
    #     gat_f = gat_features[:, :, all_indices, :]                          # (B, S, N, H)
    #     combined = torch.cat((raw_f, gat_f), dim=-1)                        # (B, S, N, 1+H)

    #     # --- 3. Pad with zeros if necessary ---
    #     pad_nodes = self.max_graph_degree - num_neighbors
    #     if pad_nodes > 0:
    #         pad = torch.zeros(
    #             (combined.size(0), combined.size(1), pad_nodes, combined.size(-1)),
    #             dtype=combined.dtype,
    #             device=combined.device
    #         )
    #         combined = torch.cat((combined, pad), dim=2)                    # (B, S, max_deg, 1+H)

    #     # --- 4. Flatten neighborhood features and concat with temporal ---
    #     flat = combined.flatten(start_dim=2)                                # (B, S, max_deg*(1+H))
    #     final = torch.cat((temporal_features, flat), dim=-1)                # (B, S, T + max_deg*(1+H))

    #     return final


    # def _gat_spatial_embedder(self, xt_filtered):
    #     sequence_out = []
    #     for t in range(xt_filtered.size(1)):
    #         #extract time step t
    #         xt = xt_filtered[:, t, :]
    #         # print(f"Time step {t}, xt shape before GAT: {xt.shape}") #torch.Size([32, 9])

    #         # Apply GAT layer
    #         xt = xt.unsqueeze(-1)
    #         xt_flattened = xt.contiguous().view(-1, 1)
    #         gat_out = self.gat_layer(xt_flattened, self.edge_index)
    #         gat_out = gat_out.view(xt.size(0), -1, self.gat_layer.out_channels)

    #         sequence_out.append(gat_out)

    #     spatial_embedder = torch.stack(sequence_out, dim=1)
    #     # print(f"Sequence out shape after GAT: {spatial_embedder.shape}") #torch.Size([32, 10, 5, 64])
    #     return spatial_embedder
    
    def forward(self, x, sensor, spatial_embedder):
        # print(f"Input x shape: {x.shape}") # Input x shape: torch.Size([32, 10, 9])
        xt_filtered = x[:, :, self.temp_dim:]  # Filter out temporal features
        # print(f"Filtered x shape: {xt_filtered.shape}")
        x_time = xt_filtered[:, :, 0:self.temp_dim]  # Extract temporal features
        # print(f"Time features shape: {x_time.shape}") #torch.Size([32, 10, 4])

        # spatial_embedder = self._gat_spatial_embedder(xt_filtered)

        tokenized_data = self._tokenizer(xt_filtered, spatial_embedder, x_time, sensor)

        # print(f"xLSTM input shape before mapping: {tokenized_data.shape}")
        mapped_x = self.input_mapper(tokenized_data)
        # print(f"xLSTM input shape: {mapped_x.shape}") #[batch, seq_len, 4+max_d*65]

        # Apply dropout to the input
        xlstm_in = self.dropout(mapped_x)

        xlstm_out = self.xlstm(xlstm_in)

        # Apply dropout to the XLSTM output before the final projection
        xlstm_output = self.dropout(xlstm_out)

        prediction = self.output_proj(xlstm_output[:, -1, :])

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