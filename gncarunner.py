import torch
from torch import nn
import pandas as pd
import numpy as np

from st_gnca.training.gncatraining import train_gnca_model
from st_gnca.dataloader.database import DataBase, BatchBuilder
from st_gnca.cellmodel.cell_model import xLSTMForecast
from xlstm import (xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig,
                     sLSTMBlockConfig, sLSTMLayerConfig, FeedForwardConfig)
from st_gnca.globalmodel.gnca import GraphCellularAutomata

print("Setting up model configuration...")
# Setup device and data types
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32
DEFAULT_PATH = 'st_gnca/'
DATA_PATH = DEFAULT_PATH + 'data/synthetic/'

# Usage example
if __name__ == "__main__":

    '''
    AINDA FALTA:
    - Mapping dos ids dos sensores PEMS03 para valores de indices 0-357
    (to utilizando syntetic data com 5 sensores pra teste)
    '''

    data = DataBase(
        edges_file=DATA_PATH + 'edge.csv',
        data_file=DATA_PATH + 'data.csv'
    )
    print("DataBase initialized.")

    batches = BatchBuilder(data, batch_size=32, sequence_len=10)
    print("BatchBuilder initialized.")

    print("Starting model's configuration...")
    hidden_dim = 64
    output_dim = 1

    temporal_emb_dim = data.temporal_features.size(1)
    value_emb_dim = 1
    max_graph_degree = data.max_graph_degree
    feature_dim = temporal_emb_dim + ((hidden_dim + 1) * (max_graph_degree+1))

    print(f"Feature Embedding Dim: {feature_dim}") # 4 (temporal_dim) + (hidden_dim+1)*max_degree = 329

    input_len = feature_dim

    print(f"Cell model initialization")
    xlstm_config = xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=3, 
                num_heads=8           # More heads for complex temporal patterns
            )
        ),
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend="vanilla", 
                num_heads=4,         # Balance capacity/compute
                conv1d_kernel_size=3
            ),
            feedforward=FeedForwardConfig(
                proj_factor=2.0,      # Wider FFN (original: 1.0)
                act_fn="gelu"
            )
        ),
        context_length=input_len,     # Match input_len
        num_blocks=4,                 # Deeper stack
        embedding_dim=hidden_dim,
        slstm_at=[1, 3]               # Add sLSTM at blocks 1 and 3
)
    cell_model = xLSTMForecast(
        input_dim= input_len,  # Each sensor and its neighbors
        output_dim=1,
        hidden_dim=64,
        edge_index=data.edge_index,
        graph=data.G,
        cfg=xlstm_config
    )

    print(f"GNCA model initialization")
    gnca = GraphCellularAutomata(
        graph=data.G,
        cell_model=cell_model,
        device=DEVICE,
        dtype=DTYPE
    )
    print("Model configuration completed.")
    print("Starting training...")

    train_gnca_model(gnca, 
                     batches.get_train_loader(), 
                     optimizer=torch.optim.AdamW(gnca.parameters(), lr=0.001), 
                     criterion=nn.MSELoss(),
                     num_epochs=1,
                     temp_dim=temporal_emb_dim,
                     device=DEVICE)
    
    #


'''
    # train_loader = batches.get_train_loader()
    # print("Train loader size:", len(train_loader)) # 573 batches of 32 sequences
    # Get the first batch
    # first_batch_X, first_batch_y = next(iter(train_loader))

    # # Print the shapes
    # print("Shape of X (input features):", first_batch_X.shape)
    # print("Shape of y (target labels):", first_batch_y.shape)

    # adj_matrix = data.get_adj_matrix()
    # print("Adjacency Matrix:", adj_matrix)
    # data_features = data._concat_features()

    # print("Data Features Shape:", data_features.shape) #([26208, 358, 5])
    # print("Number of sensors:", data.num_sensors) # 358
    # print("Number of edges:", data.num_edges) #547
    # print("Temporal Embeddings Shape:", data.temporal_features.shape) #([26208, 4])
    # print("Temporal Embeddings Example:", data.temporal_features[0]) # tensor([ 0.6760,  0.7369, -0.9985, -0.0554])
    # print("Sensor Data Shape:", data.sensor_data.shape) #([26208, 358])
    # # first row of data_features for one sensor
    # print("First row of Data Features for Sensor 0:", data_features[0, 0, :]) #[ 0.6760,  0.7369, -0.9985, -0.0554,  0.0108]
    '''