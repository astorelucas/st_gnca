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
    test_gnca_model
)
# device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
DTYPE = torch.float32
epochs = 50
DATA_PATH = 'st_gnca/data/finetuning_tests/'
DEFAULT_PATH = 'st_gnca/'

if __name__ == "__main__":
    # --- data ---
    data = DataBase(
        edges_file=DATA_PATH + 'edges_sub.csv',
        data_file=DATA_PATH + 'data_sub.csv'
    )
    batches = BatchBuilder(
        data,
        batch_size=64,
        sequence_len=12,
        horizon=3,
        val_ratio=0.2,
        train_ratio=0.6,
        device=DEVICE,
        dtype=DTYPE
    )

    # --- model config (use same hyperparams as no treinamento original) ---
    hidden_dim = 256
    gat_heads = 3
    horizon = 3
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

    # --- carregar checkpoint (OrderedDict) ---
    ckpt = torch.load('st_gnca/modularity_test/gnca_model.pth', map_location=DEVICE)

    # ckpt é OrderedDict => foi salvo via model.state_dict()
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt

    # carregar pesos; usar strict=False para inspecionar chaves faltantes/inesperadas
    load_res = gnca.load_state_dict(state, strict=False)
    print("Loaded state_dict. Missing keys:", load_res.missing_keys)
    print("Unexpected keys:", load_res.unexpected_keys)

    # preparar optimizer (somente parâmetros treináveis)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, gnca.parameters()), lr=1e-4, weight_decay=1e-5)
    criterion = nn.SmoothL1Loss(beta=0.5)

    # opcional: se o checkpoint tiver estado do optimizer, carregar
    if isinstance(ckpt, dict) and 'optimizer_state_dict' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            # mover tensores internos do optimizer para o device
            for state in optimizer.state.values():
                for k, v in list(state.items()):
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(DEVICE)
        except Exception as e:
            print("Aviso: não foi possível carregar optimizer_state_dict:", e)

    num_epochs = epochs  # já definido acima
    save_path = "gnca_finetuned.pth"

    # chamar função de treino (passar val_loader se houver)
    avg_loss, training_losses, validation_losses = train_gnca_model(
        gnca,
        batches.get_train_loader(),
        optimizer,
        criterion,
        num_epochs,
        DEVICE,
        save_path=save_path,
        val_loader=batches.get_val_loader()
    )

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
                    save_predictions_path=DEFAULT_PATH + 'results/gnca_test_results_finetuning.pth',
                    scaler=data.value_embedding.embedder
                    )

    print(results)
