import copy
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import OrderedDict
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
#from torchmetrics.regression import SymmetricMeanAbsolutePercentageError

from st_gnca.modules.common import checkpoint
from st_gnca.training.evaluate import SMAPE
from st_gnca.datasets.PEMS import PEMSBase
from st_gnca.globalmodel.gnca_old import get_timestamp
from st_gnca.embeddings.temporal import from_datetime_to_pd, from_pd_to_datetime, \
  datetime_to_str
from st_gnca.embeddings.value import ScalingTransform, build_scaler

from st_gnca.datasets.datasets import collate_fn


class FineTunningDataset(Dataset):
  def __init__(self, pems, train, nodes = None, **kwargs):
    super().__init__()

    self.pems : PEMSBase = pems
    
    if nodes is None:
      self.nodes = sorted([node for node in self.pems.G.nodes()])
    else:
      self.nodes = sorted(nodes)
    self.num_nodes = len(self.nodes)
    self.increment_type = kwargs.get('increment_type','minute')
    self.increment = kwargs.get('increment',1)
    self.steps_ahead = kwargs.get('steps_ahead',10)

    self.num_samples = self.pems.num_samples

    self.index = [k for k in range(self.num_samples - self.steps_ahead)]

    self.step = kwargs.get('step', 1000)

    self.train_split = int(train * self.num_samples) 

    self.is_validation = False

    self.start = 0
    self.end = self.num_samples

  def __getitem__(self, date):
    #print(type(date))
    if isinstance(date, pd.Timestamp):
      dt1 = date
    elif isinstance(date, datetime):
      dt1 = from_datetime_to_pd(date)
    elif isinstance(date, np.datetime64):
      dt1 = date
    elif isinstance(date, int):
      dt1 = self.pems.data['timestamp'][self.index[date]]
    else:
      raise Exception("Unknown date type: {}".format(type(date)))

    try:
      X = OrderedDict()
      X['timestamp'] = datetime_to_str(dt1)
      df1 = self.pems.data[(self.pems.data['timestamp'] == dt1)]
      for ix, node in enumerate(self.nodes):
        X[str(node)] = df1[str(node)].values[0]

      y = torch.zeros(self.num_nodes, dtype=self.pems.dtype, device=self.pems.device)
    
      dt2 = get_timestamp(dt1, self.increment_type, self.increment * self.steps_ahead)
      df2 = self.pems.data[(self.pems.data['timestamp'] == dt2)]

      for ix, node in enumerate(self.nodes):
        y[ix] = torch.tensor(df2[str(node)].values, dtype=self.pems.dtype, device=self.pems.device)

    except:
      print("ERROR!: Initial date: {}   Error date: {}".format(dt1, dt2))
     
    return X,y
  
  def train(self):
    tmp = copy.deepcopy(self)
    tmp.is_validation = False
    tmp.start = 0
    tmp.end = self.train_split - self.steps_ahead
    tmp.num_samples = self.train_split - self.steps_ahead
    return tmp

  def test(self):
    tmp = copy.deepcopy(self)
    tmp.is_validation = True
    tmp.start = self.train_split 
    tmp.end = self.num_samples - self.steps_ahead
    tmp.num_samples = self.num_samples - self.train_split - self.steps_ahead
    return tmp

  def __len__(self):
    return int((self.end - self.start)/self.step)

  def __iter__(self):
    for ct in range(self.start, self.end, self.step):
      ix = self.index[ct]
      yield self[self.pems.data['timestamp'][ix]]

  def to(self, *args, **kwargs):
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]
    self.pems = self.pems.to(*args, **kwargs)
    return self

def finetune_step(DEVICE, train, test, model, loss, mape, optim, scaler, **kwargs):
  # print(f"oi1")  # Debugging line to check keys in X

  iterations = kwargs.get('iterations',1)
  increment_type = kwargs.get('increment_type','minute')
  increment = kwargs.get('increment',1)
  batch = kwargs.get('batch',10)

  model.train()
  # print(f"oi3")  # Debugging line to check keys in X  


  errors = []
  mapes = []
  #for ct in range(batch):
  #  error = torch.tensor([0], dtype=model.dtype, device=model.device)
  #  map = torch.tensor([0], dtype=model.dtype, device=model.device)
  for X, y in tqdm(train, desc="Training batches", total=len(train)):   
    optim.zero_grad()
    i = 0
    # X = X.to(DEVICE)
    # y = y.to(DEVICE)

    y_pred = model.batch_run(initial_states = X, iterations = iterations,
                      increment = increment, increment_type = increment_type)
    print(f"Finished batch run {i}") 
    
    y_pred = scaler.denormalize(y_pred)

    error = loss(y.squeeze(), y_pred.squeeze())
    map = mape(y.squeeze(), y_pred.squeeze())

    print("y_pred shape:", y_pred.shape, "y shape:", y.shape)
    print("error shape:", error.shape, "error value:", error)
    print("y_pred device:", y_pred.device, "y device:", y.device, "error device:", error.device)
    print(f"Finished loss and mape {i}")

    print("y_pred (sample):", y_pred.flatten()[:10])
    print("y (sample):", y.flatten()[:10])

    print("Any NaN in y_pred?", torch.isnan(y_pred).any().item())
    print("Any NaN in y?", torch.isnan(y).any().item())

    print("y_pred min/max:", y_pred.min().item(), y_pred.max().item())
    print("y min/max:", y.min().item(), y.max().item())
    
    error.backward()
    print(f"Finished backward {i}")
    optim.step()
    print(f"Finished optim step {i}")
    
    print(f"Batch {i} - Error: {error.cpu().item()} - MAPE: {map.cpu().item()}")

    # Grava as métricas de avaliação
    errors.append(error.cpu().item())
    mapes.append(map.cpu().item())
    i+= 1


  ##################
  # VALIDATION
  ##################
  print("Validating model...")
  model.eval()

  errors_val = []
  mapes_val = []
  with torch.no_grad():
    #for ct in range(batch):
    #  error_val = torch.tensor([0], dtype=model.dtype, device=model.device)
    #  map_val = torch.tensor([0], dtype=model.dtype, device=model.device)
    for X,y in tqdm(test, desc="Testing batches", total=len(test)):

      #X = X.to(DEVICE)
      #y = y.to(DEVICE)

      y_pred = model.batch_run(X, iterations = iterations,
                      increment_type = increment_type, increment = increment)

      error_val = loss(y.squeeze(), y_pred.squeeze())
      map_val = mape(y.squeeze(), y_pred.squeeze())

      errors_val.append(error_val.cpu().item())
      mapes_val.append(map_val.cpu().item())

  return errors, mapes, errors_val, mapes_val

def finetune_step2(DEVICE, train, val, edge_index, edge_attr, model, loss, mape, optim, scaler, **kwargs):

    model.train()
    errors, mapes = [], []
    
    # Training Loop
    for X_raw, y_raw in tqdm(train, desc="Training batches", total=len(train)):
        optim.zero_grad()
      
        # 2. Forward pass (replaces batch_run)
        y_pred = model.batch_run2(X_raw, edge_index, edge_attr, device=DEVICE)  # (B, output_len, max_length)

        # 3. Handle denormalization if needed
        if scaler:
            y_pred = scaler.denormalize(y_pred)
            y_true = scaler.denormalize(y_raw)
        else:
            y_true = y_raw

        # 4. Calculate loss - adjust squeezing based on your shapes
        error = loss(y_true.squeeze(), y_pred.squeeze())
        current_mape = mape(y_true.squeeze(), y_pred.squeeze())
        
        # 5. Backpropagation with gradient scaling if using AMP
        # Backpropagation
        if scaler:  # Mixed precision (AMP)
            scaler.scale(error).backward()
            
            # --- Gradient Clipping ADDED HERE ---
            scaler.unscale_(optim)  # Unscale gradients before clipping (critical for AMP)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,       # Adjust based on your needs
                norm_type=2.0       # L2 norm (default)
            )
            
            scaler.step(optim)     # Step with scaled gradients
            scaler.update()
        else:
            error.backward()
            # --- Gradient Clipping for FP32 ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

        errors.append(error.detach().cpu().item())
        mapes.append(current_mape.detach().cpu().item())

    # Validation Loop
    print("Validating model...")
    model.eval()
    errors_val, mapes_val = [], []
    
    with torch.no_grad():
        for X_raw, y_raw in tqdm(val, desc="Validating batches", total=len(val)):
            # Tokenize validation batch

            y_pred = model.batch_run2(X_raw, edge_index, edge_attr, device=DEVICE)  # (B, output_len, max_length)
                        
            if scaler:
                y_pred = scaler.denormalize(y_pred)
                y_true = scaler.denormalize(y_raw)
            else:
                y_true = y_raw

            error_val = loss(y_true.squeeze(), y_pred.squeeze())
            map_val = mape(y_true.squeeze(), y_pred.squeeze())

            errors_val.append(error_val.cpu().item())
            mapes_val.append(map_val.cpu().item())

    return errors, mapes, errors_val, mapes_val

def finetune_loop(DEVICE, dataset, model, display = None, **kwargs):

  model = model.to(DEVICE)

  checkpoint_file = kwargs.get('checkpoint_file', 'modelo.pt')

  if display is None:
    from IPython import display

  batch_size = kwargs.get('batch', 10)

  fig, ax = plt.subplots(1,3, figsize=(15, 5))

  epochs = kwargs.get('epochs', 10)
  lr = kwargs.get('lr', 0.001)
  optimizer = kwargs.get('optim', optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005))
  
  # Build scaler from all y in training set
  ys = []
  for _, y in dataset.train():
      ys.append(y.cpu().numpy() if hasattr(y, 'cpu') else y)
  ys = np.stack(ys)
  scaler = ScalingTransform(ys, device=DEVICE)
  
  train_ldr = DataLoader(dataset.train(), batch_size=batch_size, shuffle=True)
  test_ldr = DataLoader(dataset.test(), batch_size=batch_size, shuffle=True)

  loss = nn.MSELoss()
  #mape = SymmetricMeanAbsolutePercentageError().to(DEVICE)
  mape = SMAPE

  error_train = []
  mape_train = []
  error_val = []
  mape_val = []

  start_time = time.time()

  best = np.inf

  for epoch in range(epochs): 
    print(f"Epoch {epoch+1}/{epochs}")
    # checkpoint(model, checkpoint_file)

    errors_train, map_train, errors_val, map_val = finetune_step(DEVICE, train_ldr, test_ldr, 
                                                                 model, loss, mape, optimizer, scaler,**kwargs)

    error_train.append(np.median(errors_train))
    mape_train.append(np.median(map_train))
    error_val.append(np.median(errors_val))
    mv = np.median(map_val)
    mape_val.append(mv)

    if mv < best:
      checkpoint(model, checkpoint_file+'BEST')
      best = mv


    display.clear_output(wait=True)
    ax[0].clear()
    ax[0].plot(error_train, c='blue', label='Train')
    ax[0].plot(error_val, c='red', label='Test')
    ax[0].legend(loc='upper left')
    ax[0].set_title("LOSS - All Epochs {} - Time: {} s".format(epoch, round(time.time() - start_time, 0)))
    ax[1].clear()
    ax[1].plot(error_train[-20:], c='blue', label='Train')
    ax[1].plot(error_val[-20:], c='red', label='Test')
    ax[1].set_title("LOSS - Last 20 Epochs".format(epoch))
    ax[1].legend(loc='upper left')
    ax[2].clear()
    ax[2].plot(mape_train[-20:], c='blue', label='Train')
    ax[2].plot(mape_val[-20:], c='red', label='Test')
    ax[2].set_title("MAPE - Last 20 Epochs".format(epoch))
    ax[2].legend(loc='upper left')
    plt.tight_layout()
    display.display(plt.gcf())

  plt.savefig(checkpoint_file+".pdf", dpi=150)

  checkpoint(model, checkpoint_file)

def finetune_loop2(DEVICE, dataset, model, display=None, **kwargs):
  model = model.to(DEVICE)

  train_ds, val_ds, test_ds = dataset

  # Initialize display if not provided
  if display is None:
      from IPython import display

  # Get parameters from kwargs with defaults
  checkpoint_file = kwargs.get('checkpoint_file', 'modelo.pt')
  batch_size = kwargs.get('batch', 10)
  epochs = kwargs.get('epochs', 10)
  lr = kwargs.get('lr', 0.001)
  
  # Initialize optimizer - fixed typo in weight_decay
  optimizer = kwargs.get('optim', optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5))
  
  # Create figure for plotting
  fig, ax = plt.subplots(1, 3, figsize=(15, 5))

  # Build scaler from training data
  print("2.1 - Build scaler from training data...")
  # scaler = build_scaler(train_ds, device=DEVICE, sample_size=50_000)
  scaler = None


  # # Create data loaders
  # train_ldr = DataLoader(dataset.train(), batch_size=batch_size, shuffle=True)
  # test_ldr = DataLoader(dataset.test(), batch_size=batch_size, shuffle=False)  # Typically don't shuffle test data


  # Define loss functions
  loss = nn.MSELoss()
  mape = SMAPE  # Make sure SMAPE is properly defined

  # Initialize tracking variables
  error_train = []
  mape_train = []
  error_val = []
  mape_val = []
  best = np.inf
  start_time = time.time()

  # Create dataloaders
  print("2.2 - Creating DataLoaders...")
  train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, collate_fn=collate_fn)
  val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, collate_fn=collate_fn)
  test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, collate_fn=collate_fn)

  # Get graph data (same for all splits)
  edge_index, edge_attr = train_ds.get_graph_data()

  for epoch in range(epochs):
      print(f"Epoch {epoch+1}/{epochs}")
      
      # Training and validation
      errors_train, map_train, errors_val, map_val = finetune_step2(
          DEVICE, train_loader, val_loader, edge_index, edge_attr ,
          model, loss, mape, optimizer, scaler, **kwargs
      )

      # Store median metrics
      error_train.append(np.median(errors_train))
      mape_train.append(np.median(map_train))
      error_val.append(np.median(errors_val))
      current_mape = np.median(map_val)
      mape_val.append(current_mape)

      # Save best model
      if current_mape < best:
          checkpoint(model, checkpoint_file+'_BEST')
          best = current_mape

      # Update plots
      display.clear_output(wait=True)
      for i, (data, color, label) in enumerate(zip(
          [(error_train, error_val), 
          (error_train[-20:], error_val[-20:]), 
          (mape_train[-20:], mape_val[-20:])],
          ['blue', 'red'],
          ['Train', 'Test']
      )):
          ax[i].clear()
          ax[i].plot(data[0], c='blue', label='Train')
          ax[i].plot(data[1], c='red', label='Test')
          ax[i].legend(loc='upper left')
          titles = [
              "LOSS - All Epochs {} - Time: {} s".format(epoch, round(time.time() - start_time, 0)),
              "LOSS - Last 20 Epochs",
              "MAPE - Last 20 Epochs"
          ]
          ax[i].set_title(titles[i])
      
      plt.tight_layout()
      display.display(plt.gcf())

  # Save final model and plot
  plt.savefig(checkpoint_file+".pdf", dpi=150)
  checkpoint(model, checkpoint_file)

  return error_train, mape_train, error_val, mape_val