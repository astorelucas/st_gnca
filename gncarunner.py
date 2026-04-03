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

    temporal_emb_dim = 12

    data = DataBase(
        edges_file=DATA_PATH + 'edges_Global.csv',
        data_file=DATA_PATH + 'data_Global.csv',
        temporal_emb_dim=temporal_emb_dim
    )
    print("DataBase initialized.")
    horizon = 12  # Predicting 12 time steps ahead
    sequence_len = 12  # Using past 36 time steps

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
    hidden_dim = 256
    gat_heads = 3
    output_dim = horizon

    temporal_emb_dim = data.temporal_features.size(1)
    value_emb_dim = 1
    feature_dim = temporal_emb_dim + ((2*(hidden_dim)*gat_heads))
    # print(f"Feature Embedding Dim: {feature_dim}") # 4 (temporal_dim) + (hidden_dim+1)*max_degree = 329

    layers = 1
    dropout = 0.15
    laplacian_comp = 20
    learning_rate = 0.001

    cell_model = LSTMForecast(
        feature_dim=feature_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        edge_index=data.edge_index,
        graph=data.G,
        num_layers=layers,
        dropout=dropout
    )

    print(f"GNCA model initialization")
    gnca = GraphCellularAutomata(
        graph=data.G,
        cell_model=cell_model,
        device=DEVICE,
        dtype=DTYPE,
        temp_dim=temporal_emb_dim,
        heads=gat_heads,
        laplacian_components=laplacian_comp,  # Number of spatial embedding components
        dropout=dropout
    )

    print("Model configuration completed.")
    print("Starting training...")
    print(f"Device available: {DEVICE}")

    avg_loss, training_losses, validation_losses = train_gnca_model(
                                    gnca, 
                                    batches.get_train_loader(), 
                                    optimizer=torch.optim.AdamW(gnca.parameters(), lr=learning_rate, weight_decay=1e-5), 
                                    criterion=SmoothL1Loss(beta=0.5),
                                    num_epochs=50,  
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
