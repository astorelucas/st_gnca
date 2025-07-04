import torch
from torch.utils.data import Dataset

import torch
import numpy as np

import copy

def self_supervised_transform(x, token_dim, max_length,value_index,NULL_SYMBOL,dtype,device):
  r = np.random.rand()
  if r >= .9:
    # Remove the cell value and keep all the neighbors
    x[0,:] = torch.full([token_dim], NULL_SYMBOL, dtype = dtype, device=device)
  elif r >= .8:
    # remove the cell value e remove all neighbor values
    x[0,value_index] = torch.tensor([NULL_SYMBOL], dtype = dtype, device=device)
    x[1:,:] = torch.full((max_length-1, token_dim),NULL_SYMBOL, dtype = dtype, device=device)
  elif r >= .7:
    # Remove the cell value
    x[0,value_index] = torch.tensor([NULL_SYMBOL], dtype = dtype, device=device)
  elif r >= .6:
    # Remove neighbor values
    x[1:,:] = torch.full((max_length-1, token_dim),NULL_SYMBOL, dtype = dtype, device=device)
  elif r >= .5:
    # Introduce random noise
    x[:,value_index] = x[:,value_index] + torch.randn(max_length, dtype = dtype, device=device)/12
  return x
  

class SensorDataset(Dataset):
  def __init__(self, name, X, y, train = 0.7,
               dtype = torch.float64, **kwargs):
    super().__init__()

    self.NULL_SYMBOL = 0

    self.behavior = kwargs.get('behavior','deterministic')

    self.num_samples = len(X)

    self.num_features = kwargs.get('num_features',0)

    self.max_length = kwargs.get('max_length',0)

    self.token_dim = kwargs.get('token_dim',0)

    self.value_index= kwargs.get('value_index',0)

    self.name = name
    self.dtype = dtype
    self.device = kwargs.get('device','cpu')

    self.X = X.to(self.dtype).to(self.device)
    self.y = y.to(self.dtype).to(self.device)

    self.indexes = torch.randperm(self.num_samples)

    self.X = self.X[self.indexes]
    self.y = self.y[self.indexes]

    self.train_split = int(train * self.num_samples)
    self.is_validation = False

  def train(self) -> Dataset:
    tmp = copy.deepcopy(self)
    tmp.is_validation = False
    tmp.X = self.X[:self.train_split].to(self.device)
    tmp.y = self.y[:self.train_split].to(self.device)
    tmp.num_samples = tmp.X.size(0)
    return tmp

  def test(self) -> Dataset:
    tmp = copy.deepcopy(self)
    tmp.is_validation = True
    tmp.X = self.X[self.train_split:].to(self.device)
    tmp.y = self.y[self.train_split:].to(self.device)
    tmp.num_samples = tmp.X.size(0)
    return tmp

  # Make this function stochastic!
  # Sometimes X will return the full information,sometimes X will miss some part of the information,
  # and sometimes will return with corrupted information (add noise)
  def __getitem__(self, index):
    if self.behavior == 'deterministic':
      return self.X[index], self.y[index]
    else:
      x = torch.clone(self.X[index])
      x = self_supervised_transform(x, self.token_dim, self.max_length, self.value_index, 
                                    self.NULL_SYMBOL,self.dtype,self.device)
      return x, self.y[index]

  def __len__(self):
    return self.num_samples

  def __iter__(self):
    for ix in range(self.num_samples):
      yield self[ix]

  def __str__(self):
    return "Dataset {}: {} attributes {} samples".format(self.name, self.num_attributes, self.num_samples)
  
  def to(self, *args, **kwargs):
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]
    self.X = self.X.to(*args, **kwargs)
    self.y = self.y.to(*args, **kwargs)
    return self
  

class AllSensorDataset(Dataset):
  def __init__(self, pems, train = 0.8, **kwargs):
    super().__init__()

    self.pems = pems

    self.max_length = pems.max_length

    self.token_dim = pems.token_dim

    self.behavior = kwargs.get('behavior','deterministic')

    self.train_pct = train

    self.train_split = int(train * self.pems.num_samples) 
    self.test_split = self.pems.num_samples - self.train_split 

    self.samples = self.pems.num_samples * self.pems.num_sensors

    self.is_validation = False

  def train(self) -> Dataset:
    tmp = copy.deepcopy(self)
    tmp.is_validation = False
    return tmp

  def test(self) -> Dataset:
    tmp = copy.deepcopy(self)
    tmp.is_validation = True
    return tmp

  def __getitem__(self, index):
    if not self.is_validation:
      train_sensor_ix = index // (self.train_split - self.pems.steps_ahead)
      train_data_ix = index % (self.train_split - self.pems.steps_ahead)
      sensor = self.pems.get_sensor(train_sensor_ix)
      X,y = self.pems.get_sample(sensor, train_data_ix)
      #print(train_sensor_ix, sensor, train_data_ix)
    else:
      train_sensor_ix = index // (self.test_split - self.pems.steps_ahead)
      train_data_ix = (index % (self.test_split - self.pems.steps_ahead)) + self.train_split
      sensor = self.pems.get_sensor(train_sensor_ix)
      #print(train_sensor_ix, sensor, train_data_ix)
      X,y = self.pems.get_sample(sensor, train_data_ix)
    
    if self.behavior == 'deterministic':
      return X,y 
    else:
      return self_supervised_transform(X, self.pems.token_dim, self.pems.max_length, 
                                       self.pems.value_index, self.pems.NULL_SYMBOL,
                                       self.pems.dtype,self.pems.device), y    
    
  def __len__(self):
    if not self.is_validation:
      return (self.train_split - self.pems.steps_ahead) * self.pems.num_sensors 
    else:
      return (self.test_split - self.pems.steps_ahead) * self.pems.num_sensors 

  def __iter__(self):
    for ix in range(len(self)):
      yield self[ix]

  def to(self, *args, **kwargs):
    self.pems = self.pems.to(*args, **kwargs)
    return self