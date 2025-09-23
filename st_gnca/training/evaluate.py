import numpy as np
import pandas as pd
import torch

def MAPE(y, y_pred):
  eps = 1e-3
  return torch.mean(torch.abs((y - y_pred) / (torch.abs(y) + eps)))

def SMAPE(y, y_pred):
  return torch.mean(2*(y - y_pred).abs() / (y.abs() + y_pred.abs() + 1e-8))

def MAE(y, y_pred):
  return torch.mean((y - y_pred).abs())

def RMSE(y, y_pred):
  return torch.sqrt(torch.mean((y - y_pred) ** 2))

def nRMSE(y, y_pred):
  # I’m considering the RSME/mean(y_true) .. so for example:
  # nRMSE = 0.05 (5%) means the average prediction error is 5% of the mean of the true values
  return RMSE(y, y_pred)/torch.mean(y)


def diff_states(state1, state2):
  keys1 = [k for k in sorted(state1.keys())]
  keys2 = [k for k in sorted(state2.keys())]
  if len(keys1) != len(keys2):
    raise ValueError("Different number of keys")
  acc = []
  for k in keys1:
    v1 = state1[k]
    v2 = state2[k]
    if isinstance(v1, (pd.Timestamp, str)):
      continue
    if isinstance(v1, torch.Tensor):
      v1 = v1.cpu().detach().numpy()[0]
    if isinstance(v2, torch.Tensor):
      v2 = v2.cpu().detach().numpy()[0]
    acc.append( np.abs(v1 - v2) )
  acc = np.array(acc)
  return np.min(acc), np.median(acc), np.mean(acc), np.std(acc), np.max(acc)


def extract_tensor(model, state):
  n = len(model.nodes)
  vals = [state[str(k)] for k in model.nodes]
  return torch.tensor(vals, device=model.device, dtype=model.dtype)

def save_training_losses_csv(training_losses, save_path):
    df = pd.DataFrame({'epoch': range(1, len(training_losses)+1), 'loss': training_losses})
    df.to_csv(save_path, index=False)
    print(f"Training losses saved to: {save_path}")
    return save_path
