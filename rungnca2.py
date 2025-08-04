from st_gnca.datasets.PEMS import PEMS03
from st_gnca.globalmodel.gnca import GraphCellularAutomata
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from st_gnca.cellmodel.cell_model import CellModel_LSTM, CellModel, CellModel_xLSTM, save_as, setup
from st_gnca.finetuning import FineTunningDataset, finetune_loop
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

STEPS_AHEAD = 12
ITERATIONS = 1

# Initialize PEMS03 dataset
pems = PEMS03(
    edges_file=DATA_PATH + 'edges.csv',
    nodes_file=DATA_PATH + 'nodes.csv',
    data_file=DATA_PATH + 'data.csv',
    device=DEVICE,
    dtype=DTYPE,
    steps_ahead=STEPS_AHEAD  
)

# Create model configuration
# # Luan - check the parameters. check activation function
# config = {
#     'num_tokens': pems.max_length,
#     'dim_token': pems.token_dim,
#     'num_transformers': 3,
#     'num_heads': 8,
#     'transformer_feed_forward': 1024,
#     'transformer_activation': nn.GELU(approximate='none'),
#     'normalization': torch.nn.LayerNorm,
#     'pre_norm': False,
#     'feed_forward': 3,
#     'feed_forward_dim': 1024,
#     'feed_forward_activation': nn.GELU(approximate='none'),
#     'device': DEVICE,
#     'dtype': DTYPE
#     }

# model = CellModel(**config)


config = {
    'num_tokens': pems.max_length,
    'dim_token': pems.token_dim,
    'hidden_size': 128,
    'num_layers': 2,
    'device': DEVICE,
    'dtype': DTYPE
    }

input_len = 12
output_len = 3
hidden_dim = 128

cfg = xLSTMBlockStackConfig(
    mlstm_block=mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(conv1d_kernel_size=3, num_heads=4)
    ),
    slstm_block=sLSTMBlockConfig(
        slstm=sLSTMLayerConfig(backend="cuda", num_heads=2, conv1d_kernel_size=3),
        feedforward=FeedForwardConfig(proj_factor=1.0, act_fn="gelu")
    ),
    context_length=input_len,
    num_blocks=3,
    embedding_dim=hidden_dim,
    slstm_at=[1]
)

model = CellModel_xLSTM(pems.num_sensors, cfg)


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
NUM_EPOCHS = 1 #20-50-100
TRAIN_SPLIT = 0.7

finetune_ds = FineTunningDataset(pems,
                                 TRAIN_SPLIT,
                                 increment_type='minutes', 
                                 increment=5, 
                                 steps_ahead=1, 
                                 step=10)

finetune_loop(DEVICE, 
              finetune_ds,
              gca, 
              iterations = 12, # 
              increment_type='minutes',
              increment=5, # 5 to 5 minutes
              epochs = NUM_EPOCHS, 
              batch = BATCH_SIZE, 
              lr = LEARNING_RATE,
              optimizer = optim.AdamW(gca.parameters(), lr=LEARNING_RATE, weight_decay=0.0005)
              )

df = evaluate(finetune_ds.test(), gca, ITERATIONS, increment_type='minutes', increment=5)
# print(df)
