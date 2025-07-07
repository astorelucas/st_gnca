from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import networkx as nx

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from tensordict import TensorDict

from st_gnca.datasets.PEMS import get_config as pems_get_config
from st_gnca.embeddings.temporal import str_to_datetime, from_datetime_to_pd
from st_gnca.cellmodel.cell_model import CellModel, get_config as cell_model_get_config, load_config as cell_model_load_config
from st_gnca.tokenizer.tokenizer import NeighborhoodTokenizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_config(model, **extra):
  cell_model_config = cell_model_get_config(model.cell_model)

  gca_config = { 
    'graph': model.graph, 
    'max_length': model.max_length,
    'token_dim': model.token_dim,
    'tokenizer': model.tokenizer,
    'cell_model': model.cell_model
  }
  cell_model_config |= gca_config 
  cell_model_config |= extra
  return cell_model_config


def load_config(config):
  return GraphCellularAutomata(graph = config.pop('graph',None), 
                   max_lengh = config.pop('max_lengh',10),
                   token_dim = config.pop('token_dim',10), 
                   tokenizer = config.pop('tokenizer',None), 
                   cell_model = config.pop('cell_model',None),
                   device = config.pop('device',None), 
                   dtype = config.pop('dtype',torch.float32), 
                   **config)


def save(to_file, pems, model, **kwargs):
    DEVICE = pems.device
    DTYPE = pems.dtype
    
    extra = {}
    extra |= pems_get_config(pems)
    extra |= kwargs
    torch.save({
        'config': get_config(model, **extra),
        "weights": model.state_dict() }, 
        to_file)
    

def setup(file, pems, DEVICE):
    saved_config = torch.load(file)
    print(saved_config['config'])
    tmp = load_config(saved_config['config']).to(DEVICE)
    tmp.load_state_dict(saved_config['weights'], strict=False)
    pems = pems.to(DEVICE)
    pems.steps_ahead = saved_config['config']['steps_ahead']
    return tmp, pems


def get_timestamp(start_ts, increment_type, increment):
  return start_ts + pd.Timedelta(increment, unit=increment_type)


def timestamp_generator(start_ts, iterations, increment_type='minutes', step=1):
  for i in range(0, iterations, step):
    yield get_timestamp(start_ts, increment_type, i)
    

class GraphCellularAutomata(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()

    self.device = kwargs.get('device',DEVICE)
    self.dtype = kwargs.get('dtype',torch.float32)

    self.graph : nx.Graph = kwargs.get('graph',None)

    self.nodes : list = sorted([k for k in self.graph.nodes()]) #why sorted?

    self.num_nodes = len(self.nodes)
    self.max_length = kwargs.get('max_length',None)
    self.token_dim = kwargs.get('token_dim',None)

    self.tokenizer : NeighborhoodTokenizer  = kwargs.get('tokenizer',None)

    self.tokenizer.graph = self.graph

    self.cell_model : nn.Module = kwargs.get('cell_model',None)

  def build_state(self, date, states) -> TensorDict:
    state = {'timestamp': date}
    for ix, node in enumerate(self.nodes):
      state[str(node)] = states[ix]
    return TensorDict(state)

  # For fine-tunning
  def forward(self, sequences, **kwargs):
    return self.cell_model.forward(sequences)
  
  def step(self, timestamp, current_state):
    tokens = torch.empty(self.num_nodes, self.max_length, self.token_dim, dtype=torch.float32, device=self.device)
    # print(f"Tokens matriz size {tokens.shape} for {self.num_nodes} nodes with max length {self.max_length} and token size {self.token_dim}")
    for ix, node in enumerate(self.nodes):
      tokens[ix, :, :] = self.tokenizer.tokenize(timestamp, current_state, node) #ta construindo uma matriz de tokens, pra cada nó
    return self.forward(tokens) 
  
  def batch_run(self, initial_states, iterations, increment_type='minute', increment=1, **kwargs) -> torch.Tensor:
    # print(f"Running batch with {len(initial_states['timestamp'])} initial states.")
    batch = len(initial_states['timestamp'])
    state_history = torch.zeros(batch, self.num_nodes, dtype=torch.float32, device=self.device)
    for ix in range(batch):
      initial_state = TensorDict({key : initial_states[key][ix] for key in initial_states.keys()})
      initial_date = str_to_datetime(initial_state['timestamp'])
      state_history[ix, :] = self.run(initial_date, initial_state, iterations, 
                                      increment_type, increment, return_type='tensor', 
                                      **kwargs)
    return state_history
  
  def run_dict(self, initial_state, iterations, increment_type='minute', increment=1, **kwargs):
    initial_date = str_to_datetime(initial_state['timestamp'])
    return self.run(initial_date, initial_state, iterations, 
                                      increment_type, increment, return_type='tensordict', 
                                      **kwargs)
  
  def run(self, initial_date, initial_state, iterations, increment_type='minute', increment=1, **kwargs) -> torch.Tensor:
    if isinstance(initial_date, datetime):
      initial_date = from_datetime_to_pd(initial_date)
    
    return_type = kwargs.get('return_type','tensordict')
    current_state = TensorDict(initial_state)
    if return_type == 'tensordict':
      state_history = []
    else:
      state_history = torch.empty(self.num_nodes, dtype=torch.float32, device=self.device)
    for ix, ts in enumerate(timestamp_generator(initial_date, iterations, increment_type, increment), start=0):
      # print(f'Running iteration {ix+1}/{iterations} at {ts}')
      result = self.step(ts, current_state) # manda 
      new_state = self.build_state(ts, result) 
      current_state = new_state

      if return_type == 'tensordict':
        state_history.append(new_state)
      else:
        state_history = result.flatten()

    # print("Finished running all iterations, returning state_history.")
    return state_history
  
  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]
    self.tokenizer = self.tokenizer.to(*args, **kwargs)
    self.cell_model = self.cell_model.to(*args, **kwargs)
    return self

  def train(self, *args, **kwargs):
    super().train(*args, **kwargs)
    self.cell_model = self.cell_model.train(*args, **kwargs)
    return self
  
  def parameters(self, recurse: bool = True):
    return self.cell_model.parameters(recurse)

    