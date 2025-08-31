import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class GraphAttentionEmbedder(nn.Module):
    def __init__(
        self,
        num_nodes,
        value_dim,
        node_feat_dim,
        temporal_dim,
        hidden_dim=64,
        heads=4
    ):
        super().__init__()

        # --- Input projection layers ---
        self.value_proj = nn.Linear(value_dim, hidden_dim)
        self.node_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.time_proj = nn.Linear(temporal_dim, hidden_dim)

        # --- Graph Attention (spatial) ---
        self.gat = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)

    def forward(self, x, edge_index, node_features, temporal_features):
        print("x shape:", x.shape)                        # expected [T, N] or [B, T, N]
        print("edge_index shape:", edge_index.shape)      # expected [2, E]
        print("node_features shape:", node_features.shape) # expected [N, node_feat_dim]
        print("temporal_features shape:", temporal_features.shape) # expected [T, temporal_dim]

        """
        Args:
            x: [B, T, N, value_dim]        Value embeddings
            edge_index: [2, E]             Graph edges
            node_features: [N, node_feat_dim]
            temporal_features: [T, temporal_dim]
        Returns:
            node_embeddings: [B, T, N, hidden_dim] 
            each e_{i,t} = node_emb_{i,t} + value_emb_{i,t} + time_emb_{i,t} (after GAT)
        """
        B, T, N, _ = x.shape

        # --- Project inputs ---
        x_proj = self.value_proj(x)  # [B, T, N, hidden_dim]
        node_proj = self.node_proj(node_features).unsqueeze(0).unsqueeze(1)  # [1,1,N,hidden_dim]
        time_proj = self.time_proj(temporal_features).unsqueeze(0).unsqueeze(2)  # [1,T,1,hidden_dim]

        # --- Sum to fuse embeddings ---
        h = x_proj + node_proj + time_proj  # [B, T, N, hidden_dim]

        # --- Graph Attention (spatial) ---
        h_flat = h.view(B*T, N, -1)
        gat_out = []
        for bt in range(B*T):
            gat_out.append(self.gat(h_flat[bt], edge_index))  # [N, hidden_dim]
        gat_out = torch.stack(gat_out, dim=0)
        gat_out = gat_out.view(B, T, N, -1)  # [B, T, N, hidden_dim]

        return gat_out  # embedding e_{i,t} for each node and timestep
