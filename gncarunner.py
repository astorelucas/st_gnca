import torch

from st_gnca.training.gncatraining import (
    train_gnca_model, 
    plot_training_loss, 
    test_gnca_model
)
from st_gnca.dataloader.database import DataBase, BatchBuilder
from st_gnca.cellmodel.cell_model import xLSTMForecast, LSTMForecast, TFTForecast
from xlstm import (xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig,
                     sLSTMBlockConfig, sLSTMLayerConfig, FeedForwardConfig)
from st_gnca.globalmodel.gnca import GraphCellularAutomata
from st_gnca.training.evaluate import HybridLoss
from datetime import datetime

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
    horizon = 12  # Predicting 1 time step ahead
    sequence_len = 24  # Using past 12 time steps

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
    hidden_dim = 64
    gat_heads = 4
    output_dim = horizon

    temporal_emb_dim = data.temporal_features.size(1)
    value_emb_dim = 1
    max_graph_degree = data.max_graph_degree
    feature_dim = temporal_emb_dim + ((2*(hidden_dim)*gat_heads))
    # print(f"Feature Embedding Dim: {feature_dim}") # 4 (temporal_dim) + (hidden_dim+1)*max_degree = 329

    # input_len = feature_dim

    print(f"Cell model initialization")
    xlstm_config = xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=4, 
                num_heads=4           # More heads for complex temporal patterns
            )
        ),
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend="cuda" if torch.cuda.is_available() else "vanilla",
                num_heads=2,         # Balance capacity/compute
                conv1d_kernel_size=4
            ),
            feedforward=FeedForwardConfig(
                proj_factor=2.0,      # Wider FFN (original: 1.0)
                act_fn="swish" # trocar pra swish
            )
        ),
        context_length=sequence_len,     # Match input_len
        num_blocks=6,                 # Deeper stack
        embedding_dim=hidden_dim,
        slstm_at=[1,3]               # Add sLSTM at blocks 1 and 3
)
    
    # cell_model = xLSTMForecast(
    #     feature_dim=feature_dim,  # Each sensor and its neighbors
    #     output_dim=output_dim,
    #     hidden_dim=hidden_dim,
    #     edge_index=data.edge_index,
    #     graph=data.G,
    #     cfg=xlstm_config
    # )

    # cell_model = LSTMForecast(
    #     feature_dim=feature_dim,
    #     output_dim=output_dim,
    #     hidden_dim=hidden_dim,
    #     edge_index=data.edge_index,
    #     graph=data.G
    # )

    cell_model = TFTForecast(
        feature_dim=feature_dim,  # Each sensor and its neighbors
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        edge_index=data.edge_index,
        graph=data.G,
        device=DEVICE
    )

    print(f"GNCA model initialization")
    gnca = GraphCellularAutomata(
        graph=data.G,
        cell_model=cell_model,
        device=DEVICE,
        dtype=DTYPE,
        temp_dim=temporal_emb_dim,
        heads=gat_heads,
        laplacian_components=10,  # Number of spatial embedding components
        dropout=0.2
    )

    print("Model configuration completed.")
    print("Starting training...")
    print(f"Device available: {DEVICE}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = DEFAULT_PATH + f'saved_models/gnca_model_{timestamp}.pth'

    avg_loss, training_losses = train_gnca_model(gnca, 
                                    batches.get_train_loader(), 
                                    optimizer=torch.optim.AdamW(gnca.parameters(), lr=0.0001, weight_decay=1e-5), 
                                    criterion=HybridLoss(alpha=0.8),
                                    num_epochs=10,  # Increased since we have early stopping
                                    device=DEVICE,
                                    return_history=True,
                                    save_path=save_path,
                                    scaler=data.value_embedding.embedder,
                                    val_loader=batches.get_val_loader())
    
    print("Training completed.")

    plot_training_loss(
        training_losses,
        save_path=DEFAULT_PATH + f'results/gnca_training_loss_{timestamp}.png',
        show=False
    )

    results = test_gnca_model(gnca, 
                    batches.get_test_loader(), 
                    temp_dim=temporal_emb_dim,
                    device=DEVICE,
                    save_predictions_path=DEFAULT_PATH + f'results/gnca_test_results_{timestamp}.pth',
                    scaler=data.value_embedding.embedder
                    )
    
    print(results)

print("Testing completed.")
