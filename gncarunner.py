import torch
from torch import nn
from torch.nn import SmoothL1Loss
import pandas as pd
import numpy as np
np.random.seed(2025)
from st_gnca.training.gncatraining import (
    train_gnca_model, 
    plot_training_loss, 
    test_gnca_model
)
from st_gnca.dataloader.database import DataBase, BatchBuilder
from st_gnca.cellmodel.cell_model import xLSTMForecast, LSTMForecast
from xlstm import (xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig,
                     sLSTMBlockConfig, sLSTMLayerConfig, FeedForwardConfig)
from st_gnca.globalmodel.gnca import GraphCellularAutomata
from st_gnca.training.evaluate import HybridLoss

import time

# Setup device and data types
DEVICE = (
    torch.device('cuda') if torch.cuda.is_available()
    else torch.device('mps') if torch.backends.mps.is_available()
    else torch.device('cpu')
)
DTYPE = torch.float32
DEFAULT_PATH = 'st_gnca/'
DATA_PATH = DEFAULT_PATH + 'data/PEMS08/'

# Usage example
if __name__ == "__main__":
    print("Setting up model configuration...")
    start_time = time.time()

    '''
    Notes:
    - The weights in the edges.csv file should be normalized
    - The data-preprocessed.csv should have the timestamp column and then the sensor columns
    - The sensor IDs in the edges.csv should match those in the data-preprocessed.csv
    - The data-preprocessed.csv should not have missing values (NaNs)
    - The data-preprocessed.csv should be in chronological order
    '''

    data = DataBase(
        edges_file=DATA_PATH + 'edges_normalized.csv',
        data_file=DATA_PATH + 'data_imputed.csv'
    )
    print("DataBase initialized.")
    horizon = 12  # Predicting 12 time steps ahead
    sequence_len = 36  # Using past 36 time steps

    batches = BatchBuilder(data, 
                           batch_size=64, 
                           sequence_len=sequence_len, 
                           horizon=horizon,
                           val_ratio=0.2,
                           train_ratio=0.6,
                           device=DEVICE,
                           dtype=DTYPE)
    
    print("BatchBuilder initialized.")

    print("Starting model's configuration...")
    hidden_dim = 96
    gat_heads = 1
    output_dim = horizon

    temporal_emb_dim = data.temporal_features.size(1)
    value_emb_dim = 1
    feature_dim = temporal_emb_dim + ((2*(hidden_dim)*gat_heads))
    # print(f"Feature Embedding Dim: {feature_dim}") # 4 (temporal_dim) + (hidden_dim+1)*max_degree = 329

    # input_len = feature_dim

#     print(f"Cell model initialization")
#     xlstm_config = xLSTMBlockStackConfig(
#         mlstm_block=mLSTMBlockConfig(
#             mlstm=mLSTMLayerConfig(
#                 conv1d_kernel_size=4, 
#                 num_heads=4           # More heads for complex temporal patterns
#             )
#         ),
#         slstm_block=sLSTMBlockConfig(
#             slstm=sLSTMLayerConfig(
#                 backend="cuda" if torch.cuda.is_available() else "vanilla",
#                 num_heads=2,         # Balance capacity/compute
#                 conv1d_kernel_size=4
#             ),
#             feedforward=FeedForwardConfig(
#                 proj_factor=2.0,      # Wider FFN (original: 1.0)
#                 act_fn="swish" # trocar pra swish
#             )
#         ),
#         context_length=sequence_len,     # Match input_len
#         num_blocks=6,                 # Deeper stack
#         embedding_dim=hidden_dim,
#         slstm_at=[1,3]               # Add sLSTM at blocks 1 and 3
# )
    
#     cell_model = xLSTMForecast(
#         feature_dim=feature_dim,  # Each sensor and its neighbors
#         output_dim=output_dim,
#         hidden_dim=hidden_dim,
#         edge_index=data.edge_index,
#         graph=data.G,
#         cfg=xlstm_config
#     )

    cell_model = LSTMForecast(
        feature_dim=feature_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        edge_index=data.edge_index,
        graph=data.G,
        num_layers=1,
        dropout=0.15
    )

    print(f"GNCA model initialization")
    gnca = GraphCellularAutomata(
        graph=data.G,
        cell_model=cell_model,
        device=DEVICE,
        dtype=DTYPE,
        temp_dim=temporal_emb_dim,
        heads=gat_heads,
        laplacian_components=20,  # Number of spatial embedding components
        dropout=0.15
    )

    print("Model configuration completed.")
    print("Starting training...")
    print(f"Device available: {DEVICE}")

    avg_loss, training_losses, validation_losses = train_gnca_model(
                                    gnca, 
                                    batches.get_train_loader(), 
                                    optimizer=torch.optim.AdamW(gnca.parameters(), lr=0.0001, weight_decay=1e-5), 
                                    criterion=SmoothL1Loss(beta=0.5),
                                    num_epochs=4,  
                                    device=DEVICE,
                                    save_path=DEFAULT_PATH + 'saved_models/gnca_model.pth',
                                    val_loader=batches.get_val_loader()
                                    )
    
    print("Training completed.")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total training time: {elapsed_time:.2f} seconds")

    plot_training_loss(
        training_losses,
        validation_losses,
        save_path=DEFAULT_PATH + 'results/gnca_training_loss.png',
        show=False
    )

    results = test_gnca_model(gnca, 
                    batches.get_test_loader(), 
                    temp_dim=temporal_emb_dim,
                    device=DEVICE,
                    save_predictions_path=DEFAULT_PATH + 'results/gnca_test_results.pth',
                    scaler=data.value_embedding.embedder
                    )
    
    print(results)

print("Testing completed.")
