import torch
from torch import nn
from st_gnca.datasets.PEMS import PEMS03
from st_gnca.cellmodel import CellModel
from st_gnca.globalmodel.gnca import GraphCellularAutomata
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

print("Setting up model configuration...")
# Setup device and data types
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32

# Define paths
DEFAULT_PATH = 'st_nca/'
DATA_PATH = DEFAULT_PATH + 'data/PEMS03/'

# Initialize PEMS03 dataset
pems = PEMS03(
    edges_file=DATA_PATH + 'edges.csv',
    nodes_file=DATA_PATH + 'nodes.csv',
    data_file=DATA_PATH + 'data.csv',
    device=DEVICE,
    dtype=DTYPE,
    steps_ahead=1  # for 5-minute ahead prediction
)

# Create model configuration
config = {
    'num_tokens': pems.max_length,
    'dim_token': pems.token_dim,
    'num_transformers': 3,
    'num_heads': 16,
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

# Create the cell model
model = CellModel(**config)

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
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
TRAIN_SPLIT = 0.6
VALIDATION_SPLIT = 0.2
# Test split will be the remaining 0.1

# Get the dataset for all sensors
dataset = pems.get_allsensors_dataset()
#dataset = pems.get_sensor_dataset(300, dtype=torch.float32, behavior='deterministic')

print("Splitting dataset...")
total_size = len(dataset)
train_size = int(TRAIN_SPLIT * total_size)
val_size = int(VALIDATION_SPLIT * total_size)
test_size = total_size - train_size - val_size

# Split the dataset
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size]
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print("Initializing optimizer and loss function...")
optimizer = optim.Adam(gca.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# Training loop
def train_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for _ , (X, y) in enumerate(tqdm(train_loader, desc="Training")):
        optimizer.zero_grad()
        output = model(X)
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
            loss = criterion(output, y)
            total_loss += loss.item()
    return total_loss / len(val_loader)

# Training history
train_losses = []
val_losses = []

# Main training loop
print("Starting training...")
for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch(gca, train_loader, optimizer, criterion)
    val_loss = validate(gca, val_loader, criterion)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print(f"Training Loss: {train_loss:.6f}")
    print(f"Validation Loss: {val_loss:.6f}")
    
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

print("Training completed!")

# Final evaluation on test set
gca.eval()
test_loss = 0
with torch.no_grad():
    for X, y in tqdm(test_loader, desc="Testing"):
        output = gca(X)
        loss = criterion(output, y)
        test_loss += loss.item()
test_loss = test_loss / len(test_loader)
print(f"Final Test Loss: {test_loss:.6f}")

# Save the final model
torch.save({
    'model_state_dict': gca.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'val_losses': val_losses,
    'test_loss': test_loss,
    'config': config
}, 'final_model.pt')
