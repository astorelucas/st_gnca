from st_gnca.datasets.PEMS import PEMS03, PEMSDataset
from st_gnca.globalmodel.gnca import GraphCellularAutomata
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from st_gnca.cellmodel.cell_model import CellModel_LSTM, CellModel, CellModel_xLSTM, xLSTMForecast, save_as, setup
from st_gnca.finetuning import FineTunningDataset, WindowedForecastDataset, finetune_loop, finetune_loop2
from st_gnca.evaluate import evaluate, diff_states

from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig, sLSTMBlockConfig, mLSTMBlockConfig, sLSTMLayerConfig, mLSTMLayerConfig, FeedForwardConfig

print("Setting up model configuration...")
# Setup device and data types
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32

# Define paths
DEFAULT_PATH = 'st_gnca/'
DATA_PATH = DEFAULT_PATH + 'data/PEMS03/'

STEPS_AHEAD = 1 # for 5-minute ahead prediction

# Initialize PEMS03 dataset
pems = PEMS03(
    edges_file=DATA_PATH + 'edges.csv',
    nodes_file=DATA_PATH + 'nodes.csv',
    data_file=DATA_PATH + 'data.csv',
    device=DEVICE,
    dtype=DTYPE,
    steps_ahead=STEPS_AHEAD  
)
"""
input_len = 1 # Sequence length t_0 -> t_1
# For example, if you want to predict the next 5 minutes based on the last hour of data, set input_len to 12 (for 5-minute intervals).
# TO-do - Pra eu colocar aqui como 12, eu tenho que ajustar o FineTunningDataset para usar input_len = 12, e output_len = 1, e steps_ahead = 1.
output_len = 1
hidden_dim = 64

xlstm_config = xLSTMBlockStackConfig(
    mlstm_block=mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(conv1d_kernel_size=3, num_heads=4)
    ),
    slstm_block=sLSTMBlockConfig(
        slstm=sLSTMLayerConfig(backend="vanilla", num_heads=2, conv1d_kernel_size=3),
        feedforward=FeedForwardConfig(proj_factor=1.0, act_fn="gelu")
    ),
    context_length=input_len,
    num_blocks=3,
    embedding_dim=hidden_dim,
    slstm_at=[1]
)

model = xLSTMForecast(
    num_nodes=358,
    input_len=input_len,         # e.g., 1 hour of 5-min intervals
    output_len=output_len,         # e.g., predict next 5 mins
    hidden_dim=hidden_dim,
    cfg=xlstm_config       # your xLSTM configuration
)

"""

input_len = 12   # 12 steps (e.g., 2 hours for 10-min intervals)
output_len = 1    # Predict next 1 hour (6 steps)
hidden_dim = 128  # Larger for GAT + XLSTM fusion

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

model = xLSTMForecast(
    token_dim=9,  # Should match your tokenizer's output dimension
    max_length=pems.max_length, # Maximum number of nodes (main + neighbors)
    output_len=output_len,
    hidden_dim=hidden_dim,
    cfg=xlstm_config,
    device="cpu"
)

# model = CellModel_xLSTM(config, 9, output_len, hidden_dim)

gca = GraphCellularAutomata(device=model.device,
                             dtype=model.dtype, 
                             graph = pems.G,
                            max_length = pems.max_length, 
                            token_dim=pems.token_dim,
                            tokenizer=pems.tokenizer, 
                            cell_model = model)

print("1 - Setting up training configuration...")
BATCH_SIZE = 512 #
LEARNING_RATE = 0.001
NUM_EPOCHS = 3 #20-50-100
TRAIN_SPLIT = 0.7

"""
finetune_ds = FineTunningDataset(pems,
                                 TRAIN_SPLIT,
                                 increment_type='minutes', 
                                 increment=5, 
                                 steps_ahead=1, 
                                 step=1)

"""

# Initialize dataset
dataset = PEMSDataset.create_splits(
    DATA_PATH+"data.csv", DATA_PATH+"edges.csv", DATA_PATH+"nodes.csv",
    input_len=12, output_len=1,
    split_ratios=(0.7, 0.2, 0.1)
)


print("2 - Starting finetuning...")
finetune_loop2(DEVICE, 
              dataset,
              gca, 
              iterations = 1, # How many time steps into the future you want to simulate/predict.
              increment_type='minutes',
              increment=5, # 5 to 5 minutes
              epochs = NUM_EPOCHS, 
              batch = BATCH_SIZE, 
              lr = LEARNING_RATE,
              optimizer = optim.AdamW(gca.parameters(), lr=LEARNING_RATE, weight_decay=0.0005)
              )

# df = evaluate(finetune_ds.test(), gca, ITERATIONS, increment_type='minutes', increment=5)