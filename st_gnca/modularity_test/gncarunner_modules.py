# This is a test file for the GNCARunner module
import sys
from pathlib import Path

# coloca o diretório raiz do repositório no sys.path (pai de "st_gnca")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from st_gnca.globalmodel.gnca import GraphCellularAutomata
from st_gnca.cellmodel.cell_model import LSTMForecast
from st_gnca.dataloader.database import DataBase, BatchBuilder
from st_gnca.training.gncatraining import (
    train_gnca_model, 
    plot_training_loss, 
    test_gnca_model,
    test_gnca_model_modules
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
DTYPE = torch.float32
epochs = 50
DATA_PATH = 'st_gnca/data/PEMS08/'
DEFAULT_PATH = 'st_gnca/'


if __name__ == "__main__":
    # --- data ---
    data = DataBase(
        edges_file=DATA_PATH + 'edges_normalized.csv',
        data_file=DATA_PATH + 'data-preprocessed.csv'
    )
    batches = BatchBuilder(
        data,
        batch_size=64,
        sequence_len=36,
        horizon=12,
        val_ratio=0.2,
        train_ratio=0.6,
        device=DEVICE,
        dtype=DTYPE
    )

    # --- model config (use same hyperparams as no treinamento original) ---
    hidden_dim = 256
    gat_heads = 3
    horizon = 12
    temporal_emb_dim = data.temporal_features.size(1)
    feature_dim = temporal_emb_dim + (2 * hidden_dim * gat_heads)
    output_dim = horizon

    cell_model = LSTMForecast(
        feature_dim=feature_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        edge_index=data.edge_index,
        graph=data.G,
        num_layers=1,
        dropout=0.15
    )

    gnca = GraphCellularAutomata(
        graph=data.G,
        cell_model=cell_model,
        device=DEVICE,
        dtype=DTYPE,
        temp_dim=temporal_emb_dim,    # preenchido corretamente
        heads=gat_heads,              # preenchido corretamente
        laplacian_components=20,
        dropout=0.15
    )
    gnca.to(DEVICE)

    print("Model configured. Loading checkpoint...")

    # # --- carregar checkpoint (OrderedDict) ---
    models_pretrained = [0, 1] #quantidade de modelos pré-treinados disponíveis
    gnca_models = []

    for model in models_pretrained:
        print(f"\n--- Loading model {model} ---")

        ckpt = torch.load(
            f'st_gnca/modularity_test/gnca_model_{model}.pth',
            map_location=DEVICE
        )

        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        else:
            state = ckpt

        load_res = gnca.load_state_dict(state, strict=False)

        print("Missing keys:", load_res.missing_keys)
        print("Unexpected keys:", load_res.unexpected_keys)

        gnca.eval()  # inferência

        gnca_models.append(gnca)


    save_path = "gnca_modularity_test.pth"
    selected_nodes = [6, 12, 39, 41, 44, 45, 47, 64, 71, 77, 103, 106, 107, 116, 145, 147, 150, 151, 158, 161, 168]

    results = test_gnca_model_modules(gnca_models, 
                                      gnca,
                                      selected_nodes,
                                      batches.get_test_loader(),
                                      temp_dim=temporal_emb_dim,
                                      device=DEVICE,
                                      save_predictions_path=DEFAULT_PATH + 'results/gnca_modularity_test.pth',
                                      scaler=data.value_embedding.embedder
                    )

    print(results)
