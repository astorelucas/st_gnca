import torch
from torch import nn
import pandas as pd
import numpy as np

from st_gnca.modules.graphattention import GraphAttentionEmbedder
from st_gnca.embeddings.spatial import SpatialEmbedding
from st_gnca.embeddings.value import ValueEmbedding

from st_gnca.datasets.PEMS import GraphTransformer

print("Setting up model configuration...")
# Setup device and data types
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32

# Define paths
DEFAULT_PATH = 'st_gnca/'
DATA_PATH = DEFAULT_PATH + 'data/PEMS03/'

# Usage example
if __name__ == "__main__":
    
    # 1. Entrar com o dataset
    data_path = DATA_PATH + "data.csv"
    edges_path = DATA_PATH + "edges.csv"
    nodes_path = DATA_PATH + "nodes.csv"
    
    nodes_df = pd.read_csv(nodes_path)
    num_nodes = len(nodes_df)
    data_df = pd.read_csv(data_path)
    dates = [pd.to_datetime(ts) for ts in data_df.iloc[:, 0].values]
    
    input_dim = 1  # Traffic flow data
    embedding_dim = 32
    k = 10
    
    # Initialize model
    model = GraphTransformer(
        num_nodes=358, 
        input_dim=1, 
        embedding_dim=64,
        dates=dates
    )
    
    # Load data and build graph
    graph_data = model.load_data(data_path, edges_path, nodes_path)
    
    # # Get sample for node 313344 at timestamp "09/01/2018 0:00"
    node_index = 313344
    timestamp = "09/01/2018 0:00"
    value_emb, node_emb, temporal_emb = model.get_sample(node_index, timestamp)

    print(f"Value embedding: {value_emb[:10]}")
    print(f"Node embedding: {node_emb[:10]}")
    print(f"Temporal embedding: {temporal_emb[:10]}")
    
    # Get combined embedding
    combined_emb = model.get_combined_embedding(node_index, timestamp)
    print(f"Combined embedding: {combined_emb[:10]}")

# 2. Criar o embedder, com parametros de entrada o G e um tempo i

# 3. Construir o Batch

# 4. Construir o tokenizer

# 5. Construir o tensor de treinamento

# 6. Loop de treinamento