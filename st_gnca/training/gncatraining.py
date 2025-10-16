import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
from st_gnca.training.evaluate import MAPE, SMAPE, MAE, RMSE, nRMSE, save_training_losses_csv
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from tqdm.auto import tqdm
import numpy as np



def train_gnca_model(gnca, train_loader, optimizer, criterion, num_epochs, device, save_path=None, return_history: bool = False, scaler=None, val_loader=None, temp_dim=None):
    """
    Train GNCA and optionally save the model state_dict to save_path after training completes.

    If return_history is True, returns (avg_loss, training_losses) where training_losses is a list
    of per-epoch average losses.
    """
    training_losses = []

    # Setup LR scheduler on validation loss if a val_loader is provided
    scheduler = None
    if val_loader is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

    early_stopping = EarlyStopping(patience=3, verbose=True, delta=0.0001, path=save_path)

    for epoch in range(num_epochs):
        gnca.train()
        total_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", leave=False):

            optimizer.zero_grad()

            # Batch X shape: torch.Size([32, 10, 358]), Batch y shape: torch.Size([32, 358])
            # Batch X shape: torch.Size([32, 12, 9]), Batch y shape: torch.Size([32, 3, 9])
            # print(f"Batch X shape: {X_batch.shape}, Batch y shape: {y_batch.shape}")
            
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            # print(f"X_batch shape: {X_batch.shape}, y_batch shape: {y_batch.shape}")

            outputs = gnca.call_model(X_batch, mode='train')
            # print(f"Outputs shape: {outputs.shape}")
            # Outputs shape: torch.Size([32, 5, 3])

            # Remove temporal features
            output_dim = outputs.shape[-3]
            # print(f"output_dim: {output_dim}")
            y_target = y_batch[..., -output_dim:]
            # print(f"y_target shape: {y_target.shape}")

            # outputs = outputs.permute(0, 2, 1)
            y_target_transposed = y_target.transpose(1, 2)
            # print(f"y_target_transposed permuted shape: {y_target_transposed.shape}")
            y_target_reshaped = y_target_transposed.unsqueeze(-1)

            # print("Example output :", outputs[0])
            # print("Example target :", y_target[0])

            # Compute loss in normalized space
            loss = criterion(outputs, y_target_reshaped)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(gnca.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
        
        epoch_loss = total_loss / n_batches if n_batches > 0 else 0.0
        training_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}")

        # Step scheduler on validation loss if available
        if scheduler is not None:
            val_loss = evaluate_gnca_model(gnca, val_loader, criterion, temp_dim, device, scaler=scaler)
            print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}")
            scheduler.step(val_loss)

        early_stopping(val_loss, gnca)

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

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
        for X_batch, y_batch in tqdm(val_loader, unit="batch", leave=False):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = gnca.call_model(X_batch, mode='val')

            # Remove temporal features
            output_dim = outputs.shape[-3]
            # print(f"output_dim: {output_dim}")
            y_target = y_batch[..., -output_dim:]
            # print(f"y_target shape: {y_target.shape}")

            # outputs = outputs.permute(0, 2, 1)
            y_target_transposed = y_target.transpose(1, 2)
            # print(f"y_target_transposed permuted shape: {y_target_transposed.shape}")
            y_target_reshaped = y_target_transposed.unsqueeze(-1)

            # Compute validation loss in normalized space
            loss = criterion(outputs, y_target_reshaped)
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
    save_training_losses_csv(training_losses, "training_losses.csv")

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
        for X_batch, y_batch in tqdm(test_loader, unit="batch", leave=False):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            # print(f"Test batch X shape: {X_batch.shape}, y shape: {y_batch.shape}")

            outputs = gnca.call_model(X_batch, mode='val') # [358, horizon]
            # print(f"Test batch outputs shape: {outputs.shape}")

            # Remove temporal features
            output_dim = outputs.shape[-3]
            # print(f"output_dim: {output_dim}")
            y_target = y_batch[..., -output_dim:]
            # print(f"y_target shape: {y_target.shape}")

            # outputs = outputs.permute(0, 2, 1)
            y_target_transposed = y_target.transpose(1, 2)
            # print(f"y_target_transposed permuted shape: {y_target_transposed.shape}")
            y_target_reshaped = y_target_transposed.unsqueeze(-1)

            # --- Denormalize both outputs and targets before metrics ---
            if scaler is not None:
                y_target_denorm = scaler.denormalize(y_target_reshaped)
                outputs_denorm = scaler.denormalize(outputs)
            else:
                y_target_denorm = y_target_reshaped
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

        results = torch.load(save_predictions_path)

        preds = results["preds"].cpu().numpy()
        targets = results["targets"].cpu().numpy()

        # automatically flatten batch & sequence dimensions, keep horizon intact
        preds = preds.reshape(-1, preds.shape[-1])
        targets = targets.reshape(-1, targets.shape[-1])

        # build dataframe with dynamic horizon
        df = pd.DataFrame({
            **{f"pred_{i}": preds[:, i] for i in range(preds.shape[1])},
            **{f"target_{i}": targets[:, i] for i in range(targets.shape[1])},
        })

        df.to_csv("results_testing_raw.csv", index=False)


    gnca.train()
    return {'metrics': metrics, 'preds': preds, 'targets': targets}



class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last improvement.
            verbose (bool): If True, prints a message for each improvement.
            delta (float): Minimum change to qualify as an improvement.
            path (str): Path to save the best model.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
