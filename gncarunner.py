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
def simulation(horizon):
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
    sequence_len = 12  # Using past 12 time steps

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

    temporal_emb_dim = 12
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
                                    num_epochs=100,  
                                    device=DEVICE,
                                    save_path=DEFAULT_PATH + f'saved_models/gnca_model_horizon_{horizon}.pth',
                                    val_loader=batches.get_val_loader()
                                    )
    
    print("Training completed.")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total training time: {elapsed_time:.2f} seconds")

    plot_training_loss(
        training_losses,
        validation_losses,
        save_path=DEFAULT_PATH + f'results/gnca_training_loss_horizon_{horizon}.png',
        show=False
    )

    results = test_gnca_model(gnca, 
                    batches.get_test_loader(), 
                    temp_dim=temporal_emb_dim,
                    device=DEVICE,
                    save_predictions_path=DEFAULT_PATH + f'results/gnca_test_results_horizon_{horizon}.pth',
                    scaler=data.value_embedding.embedder
                    )
    
    print("Testing completed.")

    metrics = results.get('metrics', {})
    mape = metrics.get('mape', float('nan'))
    smape = metrics.get('smape', float('nan'))
    mae = metrics.get('mae', float('nan'))
    rmse = metrics.get('rmse', float('nan'))
    nrmse = metrics.get('nrmse', float('nan'))
    epochs_to_stop = len(training_losses)
    if epochs_to_stop > 0:
        est_50 = (elapsed_time / epochs_to_stop) * 50
    else:
        est_50 = float('nan')

    return mape, smape, mae, rmse, nrmse, epochs_to_stop, elapsed_time, est_50

if __name__ == "__main__":
    horizons = [3, 6, 9, 12]
    all_results = []
    
    experiment_name = "wnt"
    
    for h in horizons:
        print(f"\n{'='*40}\nRunning simulation for horizon: {h}\n{'='*40}")
        mape, smape, mae, rmse, nrmse, epochs_to_stop, time_sec, est_50 = simulation(h)
        
        all_results.append({
            'Horizon': h,
            'MAPE': mape,
            'SMAPE': smape,
            'MAE': mae,
            'RSME': rmse,
            'NRMSE': nrmse,
            'epochs_to_stop': epochs_to_stop,
            'time [seconds]': time_sec,
            'Estimativa Padronizada (50 epocas)': est_50
        })
        
    
    df_results = pd.DataFrame(all_results)
    csv_path = DEFAULT_PATH + f'results/{experiment_name}_results.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"\nAll simulations completed. Results saved to {csv_path}")
