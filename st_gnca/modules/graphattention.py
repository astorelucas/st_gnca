import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F

from torch_geometric.utils import to_dense_adj, dense_to_sparse

class GraphTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1):
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels
        
        # Use Xavier/Glorot initialization for linear layers
        self.node_proj = nn.Linear(in_channels, out_channels * heads)
        torch.nn.init.xavier_uniform_(self.node_proj.weight)
        self.node_proj.bias.data.zero_()
        
        # Learnable attention parameters
        self.attn_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.attn_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        
    def forward(self, x, edge_index):
        N = x.size(0) # Number of nodes
        
        # Project node features
        x_proj = self.node_proj(x).view(N, self.heads, self.out_channels)
        
        # Calculate attention scores
        alpha_src = (x_proj * self.attn_src).sum(dim=-1) # [N, heads]
        alpha_dst = (x_proj * self.attn_dst).sum(dim=-1) # [N, heads]
        scores = alpha_src.unsqueeze(0) + alpha_dst.unsqueeze(1) # [N, N, heads]
        
        # Convert edge_index to a dense adjacency mask
        adj_dense = to_dense_adj(edge_index, max_num_nodes=N).squeeze(0) # [N, N]
        
        # DEBUG: Check for isolated nodes (nodes with no neighbors)
        node_degrees = adj_dense.sum(dim=1)
        if (node_degrees == 0).any():
            print(f"WARNING: Found {(node_degrees == 0).sum()} isolated nodes. Adding self-loops to prevent NaN.")
            # Add self-loops for isolated nodes
            idx = torch.arange(N, device=adj_dense.device)
            adj_dense[idx, idx] = 1 # Add self-loop
            node_degrees = adj_dense.sum(dim=1) # Recalculate degrees

        # Create the mask: -inf where there is NO edge, 0 where there is an edge
        # We also force self-loops to be included so a node always attends to itself.
        adj_dense_with_self_loops = adj_dense.clone()
        idx = torch.arange(N, device=adj_dense.device)
        adj_dense_with_self_loops[idx, idx] = 1  # Ensure self-attention is always possible

        # Create the mask for the attention scores
        mask = (adj_dense_with_self_loops == 0).unsqueeze(-1) # [N, N, 1]
        scores = scores.masked_fill(mask, -1e9)  # Use a large negative number instead of -inf

        # Apply stable softmax
        attn_weights = F.softmax(scores, dim=1) # [N, N, heads]
        
        # Apply attention: weighted sum of projected features
        out = torch.einsum('ijh,jhd->ihd', attn_weights, x_proj)
        out = out.reshape(N, self.heads * self.out_channels)
        
        return out


class GraphTransformerEmbedder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=1):
        super().__init__()
        self.layer1 = GraphTransformer(in_dim, hidden_dim, heads=heads)
        self.activation = nn.ReLU()
        # Initialize the final linear layer properly
        self.layer2 = nn.Linear(hidden_dim * heads, out_dim)
        torch.nn.init.xavier_uniform_(self.layer2.weight)
        self.layer2.bias.data.zero_()
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Add a print to check input (optional)
        # print(f"Input x stats: mean={x.mean().item():.4f}, std={x.std().item():.4f}, isnan={torch.isnan(x).any()}")
        x = self.layer1(x, edge_index)
        # print(f"Post-Layer1 stats: mean={x.mean().item():.4f}, std={x.std().item():.4f}, isnan={torch.isnan(x).any()}")
        x = self.activation(x)
        x = self.layer2(x)
        # print(f"Final Embedding stats: mean={x.mean().item():.4f}, std={x.std().item():.4f}, isnan={torch.isnan(x).any()}")
        return x
