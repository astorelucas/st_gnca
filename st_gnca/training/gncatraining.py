import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from st_gnca.training.evaluate import MAPE, SMAPE, MAE, RMSE, nRMSE
from tqdm.auto import tqdm

def train_gnca_model(gnca, train_loader, optimizer, criterion, num_epochs, temp_dim, device, save_path=None, return_history: bool = False, scaler=None):
    """
    Train GNCA and optionally save the model state_dict to save_path after training completes.

    If return_history is True, returns (avg_loss, training_losses) where training_losses is a list
    of per-epoch average losses.
    """
    training_losses = []
    for epoch in range(num_epochs):
        total_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", leave=False):

            optimizer.zero_grad()

            # Batch X shape: torch.Size([32, 10, 358]), Batch y shape: torch.Size([32, 358])
            # print(f"Batch X shape: {X_batch.shape}, Batch y shape: {y_batch.shape}")
            
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = gnca.call_model(X_batch, mode='train')

            # Remove temporal features
            y_target = y_batch[:, temp_dim:]
            outputs = outputs

            # --- Denormalize both outputs and targets before loss ---
            if scaler is not None:
                y_target_denorm = scaler.denormalize(y_target)
                outputs_denorm = scaler.denormalize(outputs)
            else:
                y_target_denorm = y_target
                outputs_denorm = outputs

            loss = criterion(outputs_denorm, y_target_denorm)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            
        epoch_loss = total_loss / n_batches if n_batches > 0 else 0.0
        training_losses.append(epoch_loss)
        # Optionally print the loss for this batch
        print(f"Epoch [{epoch+1}/{num_epochs}], Epoch Loss: {epoch_loss:.4f}")

    avg_loss = sum(training_losses) / len(training_losses) if len(training_losses) > 0 else 0.0

    # Save model state_dict if a path was provided
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(gnca.state_dict(), save_path)
        print(f"Model saved to: {save_path}")

    if return_history:
        return avg_loss, training_losses

    return avg_loss

def evaluate_gnca_model(gnca, val_loader, criterion, temp_dim, device, scaler=None):
    """
    Run evaluation on validation loader and return average loss.

    Args:
        gnca: model with call_model(batch) interface
        val_loader: DataLoader for validation data
        criterion: loss function (e.g. nn.MSELoss())
        temp_dim: number of temporal feature columns to drop from targets
        device: torch.device
        scaler: optional ScalingTransform for denormalization

    Returns:
        avg_loss (float)
    """
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = gnca.call_model(X_batch, mode='val')

            # match training preprocessing: drop temporal features from target
            y_target = y_batch[:, temp_dim:]

            # If target has extra dims, trim to match outputs' last dim
            if outputs.shape != y_target.shape:
                y_target = y_target[..., : outputs.shape[-1]]

            # --- Denormalize both outputs and targets before loss ---
            if scaler is not None:
                y_target_denorm = scaler.denormalize(y_target)
                outputs_denorm = scaler.denormalize(outputs)
            else:
                y_target_denorm = y_target
                outputs_denorm = outputs

            loss = criterion(outputs_denorm, y_target_denorm)
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    return avg_loss

def plot_training_loss(training_losses, save_path: str = None, show: bool = False):
    """
    Build and return a matplotlib Figure showing training loss per epoch.

    Args:
      training_losses (list[float]): loss value per epoch
      save_path (str, optional): filepath to save the figure (PNG). If None, figure is not saved.
      show (bool): if True, plt.show() will be called.

    Returns:
      matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(6,4))
    epochs = list(range(1, len(training_losses)+1))
    ax.plot(epochs, training_losses, marker='o', linestyle='-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss per Epoch')
    ax.grid(True)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Training loss plot saved to: {save_path}")

    if show:
        plt.show()

    return fig

def test_gnca_model(gnca, test_loader, temp_dim, device, save_predictions_path: str = None, scaler=None):
    """
    Run inference on test_loader and return aggregated metrics and predictions.

    Returns a dict with:
      - 'mape','smape','mae','rmse','nrmse' (floats)
      - 'preds': tensor of all predictions (cpu)
      - 'targets': tensor of all targets (cpu)

    Optionally saves predictions+targets+metrics to save_predictions_path (torch.save).
    """
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = gnca.call_model(X_batch, mode='val')

            # align with training/validation preprocessing
            y_target = y_batch[:, temp_dim:]
            if outputs.shape != y_target.shape:
                y_target = y_target[..., : outputs.shape[-1]]

            # --- Denormalize both outputs and targets before metrics ---
            if scaler is not None:
                y_target_denorm = scaler.denormalize(y_target)
                outputs_denorm = scaler.denormalize(outputs)
            else:
                y_target_denorm = y_target
                outputs_denorm = outputs

            preds_list.append(outputs_denorm.cpu())
            targets_list.append(y_target_denorm.cpu())

    preds = torch.cat(preds_list, dim=0) if preds_list else torch.empty((0,))
    targets = torch.cat(targets_list, dim=0) if targets_list else torch.empty((0,))

    if preds.numel() == 0 or targets.numel() == 0:
        metrics = {'mape': float('nan'), 'smape': float('nan'), 'mae': float('nan'),
                   'rmse': float('nan'), 'nrmse': float('nan')}
    else:
        # evaluate.py functions expect (y, y_pred)
        metrics = {
            'mape': MAPE(targets, preds).cpu().item(),
            'smape': SMAPE(targets, preds).cpu().item(),
            'mae': MAE(targets, preds).cpu().item(),
            'rmse': RMSE(targets, preds).cpu().item(),
            'nrmse': nRMSE(targets, preds).cpu().item(),
        }

    if save_predictions_path:
        os.makedirs(os.path.dirname(save_predictions_path) or ".", exist_ok=True)
        torch.save({'preds': preds, 'targets': targets, 'metrics': metrics}, save_predictions_path)
        print(f"Test predictions and metrics saved to: {save_predictions_path}")

    gnca.train()
    return {'metrics': metrics, 'preds': preds, 'targets': targets}


