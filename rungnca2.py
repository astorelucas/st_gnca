from st_gnca.datasets.PEMS import PEMS03
from st_gnca.cellmodel.cell_model import CellModel
from st_gnca.globalmodel.gnca import GraphCellularAutomata
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from st_gnca.cellmodel.cell_model import CellModel, save_as, setup
from st_gnca.finetuning import FineTunningDataset, finetune_loop
from st_gnca.evaluate import evaluate, diff_states

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
config = {
    'num_tokens': pems.max_length,
    'dim_token': pems.token_dim,
    'num_transformers': 3,
    'num_heads': 8,
    'transformer_feed_forward': 1024,
    'transformer_activation': nn.GELU(approximate='none'),
    'normalization': torch.nn.LayerNorm,
    'pre_norm': False,
    'feed_forward': 3,
    'feed_forward_dim': 1024,
    'feed_forward_activation': nn.GELU(approximate='none'),
    'device': DEVICE,
    'dtype': DTYPE
    }

model = CellModel(**config)

gca = GraphCellularAutomata(device=model.device,
                             dtype=model.dtype, 
                             graph = pems.G,
                            max_length = pems.max_length, 
                            token_dim=pems.token_dim,
                            tokenizer=pems.tokenizer, 
                            cell_model = model)

print("Setting up training configuration...")
BATCH_SIZE = 256
LEARNING_RATE = 0.001
NUM_EPOCHS = 1
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
              iterations = 12, 
              increment_type='minutes',
              increment=5,
              epochs = NUM_EPOCHS, 
              batch = BATCH_SIZE, 
              lr = LEARNING_RATE,
              optimizer = optim.AdamW(gca.parameters(), lr=LEARNING_RATE, weight_decay=0.0005)
              )

df = evaluate(finetune_ds.test(), gca, ITERATIONS, increment_type='minutes', increment=5)
# print(df)
