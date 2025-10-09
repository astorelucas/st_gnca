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

    for epoch in range(num_epochs):
        gnca.train()
        total_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", leave=False):

            optimizer.zero_grad()

            # Batch X shape: torch.Size([32, 10, 358]), Batch y shape: torch.Size([32, 358])
            # Batch X shape: torch.Size([32, 12, 9]), Batch y shape: torch.Size([32, 3, 9])
            print(f"Batch X shape: {X_batch.shape}, Batch y shape: {y_batch.shape}")
            
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            # print(f"X_batch shape: {X_batch.shape}, y_batch shape: {y_batch.shape}")

            outputs = gnca.call_model(X_batch, mode='train')
            print(f"Outputs shape: {outputs.shape}")
            # Outputs shape: torch.Size([32, 5, 3])

            # Remove temporal features
            output_dim = outputs.shape[-2]
            print(f"output_dim: {output_dim}")
            y_target = y_batch[..., -output_dim:]
            print(f"y_target shape: {y_target.shape}")

            outputs = outputs.permute(0, 2, 1)
            print(f"Outputs permuted shape: {outputs.shape}")

            # print("Example output :", outputs[0])
            # print("Example target :", y_target[0])

            # Compute loss in normalized space
            loss = criterion(outputs, y_target)
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
            val_loss = evaluate_gnca_model2(gnca, val_loader, criterion, temp_dim, device, scaler=scaler)
            print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}")
            scheduler.step(val_loss)

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
            output_dim = outputs.shape[-2]
            y_target = y_batch[..., -output_dim:]
            outputs = outputs

            outputs = outputs.permute(0, 2, 1)

            # Compute validation loss in normalized space
            loss = criterion(outputs, y_target)
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
            output_dim = outputs.shape[-2]
            y_target = y_batch[..., -output_dim:]
            outputs = outputs

            outputs = outputs.permute(0, 2, 1)
            # # align with training/validation preprocessing
            # y_target = y_batch[:, temp_dim:]
            # if outputs.shape != y_target.shape:
            #     y_target = y_target[..., : outputs.shape[-1]]

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


def evaluate_gnca_model2(
    model,
    data_loader,
    criterion,
    temp_dim,
    device,
    scaler=None,
    compute_metrics=False
):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model.call_model(X_batch, mode="eval")

            # Align temporal dimension with target
            output_dim = outputs.shape[-2]
            y_target = y_batch[..., -output_dim:]

            outputs = outputs.permute(0, 2, 1)

            loss = criterion(outputs, y_target)
            total_loss += loss.item()
            n_batches += 1

            if compute_metrics:
                # Denormalize if scaler is provided
                if scaler is not None:
                    preds = scaler.embedder.denormalize(outputs.detach().cpu())
                    targets = scaler.embedder.denormalize(y_target.detach().cpu())
                else:
                    preds = outputs.detach().cpu()
                    targets = y_target.detach().cpu()

                all_preds.append(preds.numpy())
                all_targets.append(targets.numpy())

    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0

    if compute_metrics and len(all_preds) > 0:
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        mae = mean_absolute_error(all_targets.flatten(), all_preds.flatten())
        rmse = root_mean_squared_error(all_targets.flatten(), all_preds.flatten(), squared=False)
        mape = np.mean(np.abs((all_targets.flatten() - all_preds.flatten()) / all_targets.flatten())) * 100
        smape = 100 * np.mean(2 * np.abs(all_preds.flatten() - all_targets.flatten()) / 
                             (np.abs(all_targets.flatten()) + np.abs(all_preds.flatten()) + 1e-8))

        metrics = {"mae": mae, "rmse": rmse, "mape": mape, "smape": smape}
        return avg_loss, metrics

    return avg_loss