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
from st_gnca.finetuning import FineTunningDataset, finetune_loop

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
    edges_file=DATA_PATH + 'edges_top.csv',
    nodes_file=DATA_PATH + 'nodes_top.csv',
    data_file=DATA_PATH + 'data_top.csv',
    device=DEVICE,
    dtype=DTYPE,
    steps_ahead=STEPS_AHEAD  
)

# print(f'PEMS03 max len -  {pems.max_length}')
# print(f'PEMS03 token dim -  {pems.token_dim}')

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

# config = {
#     'num_tokens': pems.max_length,
#     'dim_token': pems.token_dim,
#     'num_transformers': 3,
#     'num_heads': 16,
#     'transformer_feed_forward': 1024,
#     'transformer_activation': nn.GELU(approximate='none'),
#     'normalization': torch.nn.modules.normalization.LayerNorm,
#     'pre_norm': False,
#     'feed_forward': 3,
#     'feed_forward_dim': 1024,
#     'feed_forward_activation': nn.GELU(approximate='none'),
#     'device': DEVICE,
#     'dtype': DTYPE,
#     'steps_ahead': STEPS_AHEAD
#  }


configLSTM = {
    'num_tokens': 12, #Sequence length T
    'dim_token': pems.token_dim, #Input dim per node F
    'hidden_size': 1024,
    'lstm_depth': 3,
    'lstm_feed_forward': 1024,
    'feed_forward_activation': nn.GELU(approximate='none'),
    'output_dim': 1,  # Output dimension per node F
    'device': DEVICE,
    'dtype': DTYPE
}

# Create the cell model
model = CellModel(**config)

# cm = CellModel(num_tokens = pems.max_length, dim_token = pems.token_dim,
#                num_transformers = 2, num_heads = 8, feed_forward = 512, transformer_activation = nn.GELU(),
#                mlp = 3, mlp_dim = 512, mlp_activation = nn.GELU(), dtype = torch.float64)

''''.  '''
# Create the Graph Cellular Automata
gca = GraphCellularAutomata(
    device=model.device,
    dtype=model.dtype,
    graph=pems.G,
    max_length=pems.max_length,
    token_size=pems.token_dim,
    tokenizer=pems.tokenizer,
    cell_model=model
)

print("Setting up training configuration...")
BATCH_SIZE = 4096
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
TRAIN_SPLIT = 0.6
VALIDATION_SPLIT = 0.2
# Test split will be the remaining 0.1

finetune_ds = FineTunningDataset(pems, increment_type='hours', increment=1, steps_ahead=1, step=250, train = 0.7)

print("Initializing optimizer and loss function...")
optimizer = optim.AdamW(gca.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# Training loop
def train_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for _ , (X, y) in enumerate(tqdm(train_loader, desc="Training")):
        optimizer.zero_grad()
        output = model(X)
        y = y.view(output.shape) 
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Validation loop
def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in tqdm(val_loader, desc="Validation"):
            output = model(X)
            y = y.view(output.shape) 
            loss = criterion(output, y)
            total_loss += loss.item()
    return total_loss / len(val_loader)

# Training history
train_losses = []
val_losses = []



# --- Timing: Training ---
print("Starting training...")
if DEVICE.type == 'cuda':
    torch.cuda.synchronize()
train_start_time = time.time()
for epoch in range(NUM_EPOCHS):
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    epoch_start_time = time.time()
    train_loss = train_epoch(gca, train_loader, optimizer, criterion)
    val_loss = validate(gca, val_loader, criterion)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    epoch_end_time = time.time()
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print(f"Training Loss: {train_loss:.6f}")
    print(f"Validation Loss: {val_loss:.6f}")
    print(f"Epoch time: {epoch_end_time - epoch_start_time:.2f} seconds ({(epoch_end_time - epoch_start_time)/60:.2f} min)")
    # Save model checkpoint
    if (epoch + 1) % 10 == 0:
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': gca.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pt')
if DEVICE.type == 'cuda':
    torch.cuda.synchronize()
train_end_time = time.time()
print(f"Total training time: {train_end_time - train_start_time:.2f} seconds ({(train_end_time - train_start_time)/60:.2f} min)")

# --- Plot training and validation loss per epoch ---
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='x')
plt.title('Training and Validation Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


def compute_mae(pred, target):
    return torch.mean(torch.abs(pred - target)).item()

def compute_rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()

print("Training completed!")



# --- Timing: Test ---
if DEVICE.type == 'cuda':
    torch.cuda.synchronize()
test_start_time = time.time()
gca.eval()
test_loss = 0
mae_total = 0
rmse_total = 0
with torch.no_grad():
    for X, y in tqdm(test_loader, desc="Testing"):
        output = gca(X)
        y = y.view(output.shape)
        loss = criterion(output, y)
        test_loss += loss.item()
        mae_total += compute_mae(output, y)
        rmse_total += compute_rmse(output, y)
num_batches = len(test_loader)
test_loss = test_loss / num_batches
test_mae = mae_total / num_batches
test_rmse = rmse_total / num_batches
if DEVICE.type == 'cuda':
    torch.cuda.synchronize()
test_end_time = time.time()
print(f"Final Test Loss (MSE): {test_loss:.6f}")
print(f"Final Test MAE: {test_mae:.6f}")
print(f"Final Test RMSE: {test_rmse:.6f}")
print(f"Total Test time: {test_end_time - test_start_time:.2f} seconds ({(test_end_time - test_start_time)/60:.2f} min)")

# --- Plot predictions vs real values for a small period ---
# Get a small batch from the test set
gca.eval()
with torch.no_grad():
    X_batch, y_batch = next(iter(test_loader))
    output_batch = gca(X_batch)

# Select the first 100 samples (or less if batch is smaller)
n_samples = min(100, y_batch.shape[0])
real = y_batch[:n_samples].cpu().numpy().flatten()
pred = output_batch[:n_samples].cpu().numpy().flatten()

plt.figure(figsize=(12, 5))
plt.plot(real, label='Real', marker='o')
plt.plot(pred, label='Predicted', marker='x')
plt.title('Predicted vs Real Values (First 100 samples of test batch)')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.legend()
plt.tight_layout()
plt.show()

# Save the final model and metrics
torch.save({
    'model_state_dict': gca.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'val_losses': val_losses,
    'test_loss': test_loss,
    'test_mae': test_mae,
    'test_rmse': test_rmse,
    'config': config
}, 'final_model.pt')
