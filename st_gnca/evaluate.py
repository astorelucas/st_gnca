import numpy as np
import pandas as pd
import torch

from st_gnca.embeddings.temporal import str_to_datetime, from_datetime_to_pd
from st_gnca.globalmodel.gnca import get_timestamp

def MAPE(y, y_pred):
  return torch.mean((y - y_pred).abs() / (y.abs() + 1e-8))

def SMAPE(y, y_pred):
  return torch.mean(2*(y - y_pred).abs() / (y.abs() + y_pred.abs() + 1e-8))

def MAE(y, y_pred):
  return torch.mean((y - y_pred).abs())

def RMSE(y, y_pred):
  return torch.sqrt(torch.mean((y - y_pred) ** 2))

def nRMSE(y, y_pred):
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


def evaluate(dataset, gca, steps_ahead, increment_type='minutes', increment=5):
  gca.eval()
  columns = ['timestamp','mape','mae','rmse','nrmse']
  rows = []
  with torch.no_grad():
    for ix in range(len(dataset) - increment * steps_ahead):
      X,y = dataset[ix]
      p = gca.run(str_to_datetime(X['timestamp']), X, iterations=steps_ahead, 
                  increment_type=increment_type, increment=increment, 
                  return_type='tensor').detach()
      row = [get_timestamp(str_to_datetime(X['timestamp']),increment_type, increment * steps_ahead)]
      
      row.extend([
        SMAPE(y, p).cpu().item(), 
        MAE(y, p).cpu().item(), 
        RMSE(y, p).cpu().item(), 
        nRMSE(y, p).cpu().item()]
        )
      rows.append(row)
  return pd.DataFrame(rows, columns=columns)


def residuals(dataset, gca, steps_ahead, increment_type='minutes', increment=5):
  gca.eval()
  columns = ['timestamp','residual']
  rows = []
  with torch.no_grad():
    for ix in range(len(dataset) - increment * steps_ahead):
      X,y = dataset[ix]
      p = gca.run(str_to_datetime(X['timestamp']), X, iterations=steps_ahead, 
                  increment_type=increment_type, increment=increment, 
                  return_type='tensor').detach()
      row = [get_timestamp(str_to_datetime(X['timestamp']),increment_type, increment * steps_ahead)]
      
      row.append(torch.median(y - p).cpu().item())
      rows.append(row)
  return pd.DataFrame(rows, columns=columns)