import torch
from torch import nn
from torch.nn import SmoothL1Loss
import pandas as pd
import numpy as np

np.random.seed(2025)
from st_gnca.training.gncatraining import (
    train_gnca_model,
    plot_training_loss,
    test_gnca_model,
)
from st_gnca.dataloader.database import DataBase, BatchBuilder
from st_gnca.cellmodel.cell_model import xLSTMForecast, LSTMForecast
from xlstm import (
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)
from st_gnca.globalmodel.gnca import GraphCellularAutomata
from st_gnca.training.evaluate import HybridLoss
from st_gnca.modules.common import load_yaml_config
import time
from pathlib import Path
import wandb
from datetime import datetime

# Setup device and data types
DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
)
DTYPE = torch.float32
DEFAULT_PATH = "./st_gnca/"

class DummyRunner:
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass

    def log_model(self, *args, **kwargs):
        pass
    
    def log_artifact(self, *args, **kwargs):
        pass
    
    def finish(self, *args, **kwargs):
        pass
     
# Usage example
if __name__ == "__main__":
    print("Setting up model configuration...")
    start_time = time.time()

    """
    Notes:
    - The weights in the edges.csv file should be normalized
    - The data-preprocessed.csv should have the timestamp column and then the sensor columns
    - The sensor IDs in the edges.csv should match those in the data-preprocessed.csv
    - The data-preprocessed.csv should not have missing values (NaNs)
    - The data-preprocessed.csv should be in chronological order
    """

    print("Loading model configuration from yaml...")
    config = load_yaml_config("config.yaml")
    dataset = config["forecasting"]["dataset"]
    data_path = Path(DEFAULT_PATH) / 'data' / dataset
    print("Initializing DataBase...")
    data = DataBase(
        edges_file=str(data_path / "edges.csv"),
        data_file=str(data_path / "data.csv"),
    )
    print("DataBase initialized.")

    batches = BatchBuilder(
        data,
        batch_size=config["forecasting"]["batch_size"],
        sequence_len=config["forecasting"]["sequence_len"],
        horizon=config["forecasting"]["horizon"],
        val_ratio=0.2,
        train_ratio=0.6,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("BatchBuilder initialized.")

    run = DummyRunner()
    if config['wandb'].get("enabled", False):
        print("Setting up Weights and Biases Project")
        current_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run = wandb.init(
            entity=config["wandb"]["entity"],
            project=config["wandb"]["project"],
            config={k: v for k, v in config.items() if k != "wandb"},
            name=f"gnca_forecasting_{dataset.lower()}_horizon-{config["forecasting"]["horizon"]}_{current_ts}",
        )

    print("Starting model's configuration...")
    hidden_dim = config["lstm"]["hidden_dim"]
    gat_heads = config["gat"]["heads"]
    output_dim = config["forecasting"]["horizon"]

    temporal_emb_dim = data.temporal_features.size(1)
    value_emb_dim = 1
    feature_dim = temporal_emb_dim + ((2 * (hidden_dim) * gat_heads))
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
        num_layers=config["lstm"]["layers"],
        dropout=config["lstm"]["dropout"],
    )

    print(f"GNCA model initialization")
    gnca = GraphCellularAutomata(
        graph=data.G,
        cell_model=cell_model,
        device=DEVICE,
        dtype=DTYPE,
        hidden_dim=config["gat"]["hidden_dim"],
        temp_dim=temporal_emb_dim,
        heads=gat_heads,
        laplacian_components=config["gat"][
            "laplacian_components"
        ],  # Number of spatial embedding components
        dropout=config["gat"]["dropout"],
    )
    if config['wandb'].get("enabled", False):
        wandb.watch(gnca, log="all")

    print("Starting training...")
    avg_loss, training_losses, validation_losses = train_gnca_model(
        gnca,
        batches.get_train_loader(),
        optimizer=torch.optim.AdamW(gnca.parameters(), lr=0.0001, weight_decay=1e-5),
        criterion=SmoothL1Loss(beta=0.5),
        num_epochs=config["forecasting"]["epochs"],
        device=DEVICE,
        save_path=DEFAULT_PATH + "saved_models/gnca_model.pth",
        val_loader=batches.get_val_loader(),
        run=run
    )

    print("Training completed.")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total training time: {elapsed_time:.2f} seconds")

    fig = plot_training_loss(
        training_losses,
        validation_losses,
        save_path=DEFAULT_PATH + "results/gnca_training_loss.png",
        show=False,
    )
    
    run.log({"training_validation_loss_plot": fig})

    results = test_gnca_model(
        gnca,
        batches.get_test_loader(),
        temp_dim=temporal_emb_dim,
        device=DEVICE,
        save_predictions_path=DEFAULT_PATH + "results/gnca_test_results.pth",
        scaler=data.value_embedding.embedder,
    )
    print(results)
    
    run.log({"test_metrics": results['metrics']})
    
    artifact = wandb.Artifact(name="gnca_predictions_and_targets", type="dataset")
    artifact.add_file('results_testing_raw.csv')
    run.log_artifact(artifact)
    
    run.finish()

print("Testing completed.")
