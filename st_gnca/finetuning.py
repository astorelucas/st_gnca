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

from st_gnca.common import checkpoint
from st_gnca.evaluate import SMAPE
from st_gnca.datasets.PEMS import PEMSBase
from st_gnca.globalmodel.gnca import get_timestamp
from st_gnca.embeddings.temporal import from_datetime_to_pd, from_pd_to_datetime, \
  datetime_to_str


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



def finetune_step(DEVICE, train, test, model, loss, mape, optim, **kwargs):

  iterations = kwargs.get('iterations',1)
  increment_type = kwargs.get('increment_type','minute')
  increment = kwargs.get('increment',1)
  batch = kwargs.get('batch',10)

  model.train()

  errors = []
  mapes = []
  #for ct in range(batch):
  #  error = torch.tensor([0], dtype=model.dtype, device=model.device)
  #  map = torch.tensor([0], dtype=model.dtype, device=model.device)
  for X, y in tqdm(train, desc="Training batches", total=len(train)):   
    optim.zero_grad()
    print("X shape: ", X.shape, "y shape: ", y.shape)
    print("X: ", X)
    print("y: ", y)

    #X = X.to(DEVICE)
    #y = y.to(DEVICE)

    y_pred = model.batch_run(initial_states = X, iterations = iterations,
                      increment = increment, increment_type = increment_type)

    error = loss(y.squeeze(), y_pred.squeeze())
    map = mape(y.squeeze(), y_pred.squeeze())

    error.backward()
    optim.step()

    # Grava as métricas de avaliação
    errors.append(error.cpu().item())
    mapes.append(map.cpu().item())


  ##################
  # VALIDATION
  ##################

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

# finetune_loop(DEVICE, 
#               finetune_ds,
#               gca, 
#               iterations = 12, 
#               increment_type='minutes',
#               increment=5,
#               epochs = NUM_EPOCHS, 
#               batch = BATCH_SIZE, 
#               lr = LEARNING_RATE,
#               optimizer = optim.AdamW(gca.parameters(), lr=LEARNING_RATE, weight_decay=0.0005)
#               )

# finetune_ds = FineTunningDataset(
#                       pems, 
#                       increment_type='hours', 
#                       increment=1, 
#                       steps_ahead=1, 
#                       step=250, 
#                       train = 0.7)

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
    checkpoint(model, checkpoint_file)

    errors_train, map_train, errors_val, map_val = finetune_step(DEVICE, train_ldr, test_ldr, 
                                                                 model, loss, mape, optimizer, **kwargs)

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