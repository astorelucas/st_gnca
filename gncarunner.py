import torch
from torch import nn
import pandas as pd
import numpy as np

from st_gnca.training.gncatraining import (
    train_gnca_model, 
    plot_training_loss, 
    evaluate_gnca_model,
    test_gnca_model
)
from st_gnca.dataloader.database import DataBase, BatchBuilder
from st_gnca.cellmodel.cell_model import xLSTMForecast
from xlstm import (xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig,
                     sLSTMBlockConfig, sLSTMLayerConfig, FeedForwardConfig)
from st_gnca.globalmodel.gnca import GraphCellularAutomata
from st_gnca.embeddings.value import ScalingTransform

print("Setting up model configuration...")
# Setup device and data types
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32
DEFAULT_PATH = 'st_gnca/'
DATA_PATH = DEFAULT_PATH + 'data/synthetic/'

# Usage example
if __name__ == "__main__":

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

    # print(f"Feature Embedding Dim: {feature_dim}") # 4 (temporal_dim) + (hidden_dim+1)*max_degree = 329

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

    scaler = ScalingTransform()

    avg_loss, training_losses = train_gnca_model(gnca, 
                                    batches.get_train_loader(), 
                                    optimizer=torch.optim.AdamW(gnca.parameters(), lr=0.001), 
                                    criterion=nn.MSELoss(),
                                    num_epochs=2,
                                    temp_dim=temporal_emb_dim,
                                    device=DEVICE,
                                    return_history=True,
                                    save_path=DEFAULT_PATH + 'saved_models/gnca_model.pth',
                                    scaler=scaler)
    
    print("Training completed.")

    evaluate_gnca_model(gnca, 
                        batches.get_val_loader(), 
                        criterion=nn.MSELoss(),
                        temp_dim=temporal_emb_dim,
                        device=DEVICE,
                        scaler=scaler)
    print("Evaluation completed.")

    plot_training_loss(
        training_losses,
        save_path=DEFAULT_PATH + 'results/gnca_training_loss.png',
        show=True
    )

    results = test_gnca_model(gnca, 
                    batches.get_test_loader(), 
                    temp_dim=temporal_emb_dim,
                    device=DEVICE,
                    save_predictions_path=DEFAULT_PATH + 'results/gnca_test_results.pth',
                    scaler=scaler
                    )
    
    print(results)
    print("Testing completed.")
