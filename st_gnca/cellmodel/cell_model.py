import torch
from torch import nn
import numpy as np

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
        self.graph = kwargs.get('graph', None)
        self.max_graph_degree = max(dict(self.graph.degree()).values())
        self.temp_dim = kwargs.get('temp_dim', 4)  # Default temporal embedding dimension

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # GAT Layer
        self.gat_layer = GATConv(
                    in_channels=1,
                    out_channels=hidden_dim
                ).to(dtype=dtype)
        
        self.input_mapper = nn.Linear(input_dim, hidden_dim).to(dtype=dtype)
        
        # XLSTM Block Stack
        self.xlstm = xLSTMBlockStack(cfg).to(dtype=dtype)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim).to(dtype=dtype)

        # Ensure all parameters are on correct device and dtype
        self.to(device=device, dtype=dtype)


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

        # --- 1. Identify the nodes in the neighborhood ---
        neighbor_indices1 = self.edge_index[1][self.edge_index[0] == target_sensor_idx].cpu().numpy()
        neighbor_indices2 = self.edge_index[0][self.edge_index[1] == target_sensor_idx].cpu().numpy()
        all_neighbors = list(neighbor_indices1) + list(neighbor_indices2)
        all_neighbors = np.unique(all_neighbors)  # Remove duplicates if any

        assert len(all_neighbors) <= self.max_graph_degree and len(all_neighbors) > 0, \
            f"Sensor {target_sensor_idx} has {len(all_neighbors)} neighbors, which exceeds max graph degree {self.max_graph_degree} or is zero."

        all_indices = np.append(target_sensor_idx, all_neighbors)
        # print(f"All indices for sensor {target_sensor_idx}: {all_indices}")

        # --- 2. Filter all input tensors for the neighborhood ---
        raw_features_filtered = raw_features[:, :, all_indices]
        # print(f"Raw features filtered shape: {raw_features_filtered.shape}") # Expected shape: (B, S, num_nodes)
        gat_features_filtered = gat_features[:, :, all_indices, :]
        # print(f"GAT features filtered shape: {gat_features_filtered.shape}") # Expected shape: (B, S, num_nodes, H)
        
        num_neighborhood_nodes = len(all_indices)
        
        # --- 3. Perform the feature aggregation (without padding yet) ---
        raw_features_reshaped = raw_features_filtered.unsqueeze(-1)
        # n_temporal = temporal_features.size(-1) # Number of temporal features = 4 dim
        # temporal_features_broadcast = temporal_features.unsqueeze(2).repeat(1, 1, num_neighborhood_nodes, 1)
        # print(f"Temporal features broadcast shape: {temporal_features_broadcast.shape}") # Expected shape: (B, S, num_nodes, T)

        combined_tensor = torch.cat(
            (raw_features_reshaped, gat_features_filtered),
            dim=-1
        )

        # print(f"Combined tensor shape before padding: {combined_tensor.shape}") # Expected shape: (B, S, num_nodes, 1 + T + H)

        # print(f"All neighbors for sensor {target_sensor_idx}: {all_neighbors}")
        # --- 4. Pad with zeros if necessary ---
        # print(f"Number of neighborhood nodes: {len(all_neighbors)}, Max graph degree: {self.max_graph_degree}")
        if (len(all_neighbors)) < self.max_graph_degree:
            padding_size = self.max_graph_degree - (len(all_neighbors)) 
            # print(f"Padding size: {padding_size}")
            total_features = combined_tensor.size(-1)
            
            # Create a padding tensor filled with zeros
            padding = torch.zeros(
                (combined_tensor.size(0), combined_tensor.size(1), padding_size, total_features),
                dtype=combined_tensor.dtype,
                device=combined_tensor.device
            )
            
            # Concatenate the original tensor with the padding tensor
            combined_tensor = torch.cat((combined_tensor, padding), dim=2)
        # print(f"Combined tensor shape after padding: {combined_tensor.shape}") # Expected shape: (B, S, max_degree+1, 1 + T + H)
        
        flat_tensor = combined_tensor.view(
                combined_tensor.size(0),
                combined_tensor.size(1),
                -1
            )
        
        final = torch.cat((temporal_features, flat_tensor), dim=-1) # Concatenate temporal features
        # print(f"Final flattened tensor shape: {final.shape}") # Expected shape: (B, S, (max_degree+1)*(1 + T + H))

        return final

    def _gat_spatial_embedder(self, xt_filtered):
        sequence_out = []
        for t in range(xt_filtered.size(1)):
            #extract time step t
            xt = xt_filtered[:, t, :]
            # print(f"Time step {t}, xt shape before GAT: {xt.shape}") #torch.Size([32, 9])

            # Apply GAT layer
            xt = xt.unsqueeze(-1)
            xt_flattened = xt.contiguous().view(-1, 1)
            gat_out = self.gat_layer(xt_flattened, self.edge_index)
            gat_out = gat_out.view(xt.size(0), -1, self.gat_layer.out_channels)

            sequence_out.append(gat_out)

        spatial_embedder = torch.stack(sequence_out, dim=1)
        # print(f"Sequence out shape after GAT: {spatial_embedder.shape}") #torch.Size([32, 10, 5, 64])
        return spatial_embedder
    
    def forward(self, x, sensor):
        # print(f"Input x shape: {x.shape}") # Input x shape: torch.Size([32, 10, 9])
        xt_filtered = x[:, :, self.temp_dim:]  # Filter out temporal features
        # print(f"Filtered x shape: {xt_filtered.shape}")
        x_time = xt_filtered[:, :, 0:self.temp_dim]  # Extract temporal features
        # print(f"Time features shape: {x_time.shape}") #torch.Size([32, 10, 4])

        spatial_embedder = self._gat_spatial_embedder(xt_filtered)

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