import torch
from torch import nn
import pandas as pd
import numpy as np

from st_gnca.modules.graphattention import GraphTransformerEmbedder, GraphTransformer
from st_gnca.embeddings.spatial import SpatialEmbedding
from st_gnca.embeddings.value import ValueEmbedding

from st_gnca.datasets.PEMS import build_data

from torch_geometric.data import Data


print("Setting up model configuration...")
# Setup device and data types
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32



# Usage example
if __name__ == "__main__":


    data = build_data()

    print("\n" + "="*50)
    print("DATA SANITY CHECKS")
    print("="*50)
    print(f"Data features shape: {data.x.shape}")
    print(f"Data features - Mean: {data.x.mean().item():.4f}, Std: {data.x.std().item():.4f}")
    print(f"Data features - Min: {data.x.min().item():.4f}, Max: {data.x.max().item():.4f}")
    print(f"Data features - NaN values: {torch.isnan(data.x).sum().item()}")
    print(f"Data features - Inf values: {torch.isinf(data.x).sum().item()}")

    # 4.2 Initialize model
model = GraphTransformerEmbedder(
    in_dim=data.num_features, # Use the actual feature dimension
    hidden_dim=16,
    out_dim=8,
    heads=2
)

# 4.3 A tiny forward pass test with a subset of data
print("\n" + "="*50)
print("RUNNING A SMALL SANITY CHECK")
print("="*50)
test_x = data.x[:5] # Test on first 5 nodes
test_edge_index = data.edge_index # Use all edges, the layer will mask automatically

# Test just the first layer
test_layer = GraphTransformer(data.num_features, 16, heads=2)
with torch.no_grad():
    test_output = test_layer(test_x, test_edge_index)
print(f"Test output shape: {test_output.shape}")
print(f"Test output - NaN values: {torch.isnan(test_output).sum().item()}")

# 4.4 Now run the full model
if torch.isnan(test_output).sum().item() == 0:
    print("\n" + "="*50)
    print("GENERATING FINAL EMBEDDINGS")
    print("="*50)
    model.eval()
    with torch.no_grad():
        node_embeddings = model(data)

    print("\nGenerated Node Embeddings:")
    print(f"Shape: {node_embeddings.shape}")
    print(f"Embeddings contain NaN: {torch.isnan(node_embeddings).any().item()}")
    print(f"Embeddings contain Inf: {torch.isinf(node_embeddings).any().item()}")
    print(f"Embedding stats - Mean: {node_embeddings.mean().item():.4f}, Std: {node_embeddings.std().item():.4f}")

    if not torch.isnan(node_embeddings).any():
        print("\nSuccess! Embeddings for the first 5 nodes:")
        print(node_embeddings[:5])
    else:
        print("\nStill getting NaN. Let's try a simpler approach.")
else:
    print("\nThe test failed. The issue is likely in the data or the graph structure.")