
import torch
from torch import nn
from tensordict import TensorDict

activations = {
  'ReLU': nn.ReLU(), 
  'GELU': nn.GELU(), 
  'GLU': nn.GLU(),
  'SELU': nn.SELU()
}

normalizations = {
  'LayerNorm': nn.LayerNorm, 
  'RMSNorm': nn.RMSNorm,
  'BatchNorm': nn.BatchNorm1d,
  'GroupNorm': nn.GroupNorm
}

dtypes = {
  'torch.int8': torch.int8,
  'torch.int16': torch.int16,
  'torch.int32': torch.int32,
  'torch.int64': torch.int64,
  'torch.int': torch.int,
  'torch.long': torch.long,
  'torch.float': torch.float,
  'torch.float16': torch.float16,
  'torch.float32': torch.float32,
  'torch.float64': torch.float64
}


def get_device():
  return 'cuda' if torch.cuda.is_available() else 'cpu'

def checkpoint(model, file):
  torch.save(model.state_dict(), file)

def checkpoint_all(file, **kwargs):
  torch.save({k: v for k,v in kwargs.items() }, file)

def resume(model, file):
  model.load_state_dict(torch.load(file, weights_only=True, map_location=torch.device(get_device())))

def resume_all(model, optimizer, file):
  checkpoint = torch.load( file, map_location=torch.device(get_device()))
  model.load_state_dict(checkpoint['model'])
  optimizer.load_state_dict(checkpoint['optim'])


class TensorDictDataframe:
  def __init__(self,**kwargs):
    self.dtype = kwargs.get('dtype',torch.float64)
    self.device = kwargs.get('device','cpu')

    df = kwargs.get('numeric_df',None)

    vals = {}
    self.numeric_columns = []
    for ix, column in enumerate(df.columns):
      vals[column] = torch.tensor(df[column].values, dtype=self.dtype) 
      self.numeric_columns.append(column)
      self.length = len(df[column].values)

    self.numeric_data = TensorDict(vals).to(self.device)

    del(df)
    del(vals)

    df = kwargs.get('nonnumeric_df',None)

    vals = {}
    self.nonnumeric_columns = []
    for ix, column in enumerate(df.columns):
      vals[column] = df[column].values
      self.nonnumeric_columns.append(column)

    self.nonnumeric_data = TensorDict(vals).to(self.device)

    del(df)
    del(vals)

  def __getitem__(self, index):
    if isinstance(index, str):
      if index in self.numeric_columns:
        return self.numeric_data[index]
      elif  index in self.nonnumeric_columns:
        return self.nonnumeric_data[index]
      else:
        raise IndexError()
    elif isinstance(index, int):
      tmp = {}
      for col in self.nonnumeric_columns:
        tmp[col] = self.nonnumeric_data[col][index]
      for col in self.numeric_columns:
        tmp[col] = self.numeric_data[col][index]
      return TensorDict(tmp)
    elif isinstance(index, (list, tuple, set)):
      if len(index) > 2:
        raise IndexError()
      
      col, ix = index
      if col in self.numeric_columns:
        return self.numeric_data[col][ix]
      elif  col in self.nonnumeric_columns:
        return self.nonnumeric_data[col][ix]
      else:
        raise IndexError()

  def __len__(self):
    return self.length

  def __iter__(self):
    for ix in range(self.length):
      yield self[ix]

  def to(self, *args, **kwargs):
    if isinstance(args[0], str):
      self.device = args[0]
      self.numeric_data = self.numeric_data.to(*args, **kwargs)
      self.nonnumeric_data = self.nonnumeric_data.to(*args, **kwargs)
    return self