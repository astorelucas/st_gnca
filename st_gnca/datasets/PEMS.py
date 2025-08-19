import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import OrderedDict

import torch

from st_gnca.embeddings.temporal import TemporalEmbedding, SinusoidalTemporalEncoding, to_pandas_datetime
from st_gnca.embeddings.spatial import SpatialEmbedding
from st_gnca.embeddings.value import ValueEmbedding
from st_gnca.tokenizer.tokenizer import NeighborhoodTokenizer

from st_gnca.common import TensorDictDataframe

from st_gnca.datasets.datasets import SensorDataset, AllSensorDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_config(pems):
  return {
    'steps_ahead': pems.steps_ahead,
    'value_embedding_type': pems.tokenizer.value_embedder.type
  }


class PEMSBase:

    def __init__(self,**kwargs):

      self.dtype = kwargs.get('dtype',torch.float64)
      self.device = kwargs.get('device',DEVICE)

      self.steps_ahead = kwargs.get('steps_ahead',1)

      edges = pd.read_csv(kwargs.get('edges_file','edges.csv'), engine='pyarrow')

      # Create the graph
      self.G=nx.Graph()
      for row in edges.iterrows():
        self.G.add_edge(int(row[1]['source']),int(row[1]['target']), weight=row[1]['weight'])

      del(edges)

      self.data = pd.read_csv(kwargs.get('data_file','data.csv'), engine='pyarrow')
      self.data['timestamp'] = to_pandas_datetime(self.data['timestamp'].values)

      self.value_embedder = ValueEmbedding(torch.tensor(self.data[self.data.columns[1:]].values,
                                                        dtype=self.dtype, device=self.device),
                                                        # value_embedding_type='scaling',
                                                        **kwargs)

      self.latlon = kwargs.get("latlon",True)

      if self.latlon:

        laplacian_components = 2

        nodes = pd.read_csv(kwargs.get('nodes_file','nodes.csv'), engine='pyarrow')

        coordinates = {}

        for ix, node in enumerate(self.G.nodes()):

            _, lat, lon = nodes[nodes['sensor'] == node].values[0]

            coordinates[node] = {'lat': lat, 'lon': lon }

        nx.set_node_attributes(self.G, coordinates)

        del(nodes)

      else:

        laplacian_components = 4

      # TO-DO - Use Graph Attention implementation of spatial embedding
      self.node_embeddings = SpatialEmbedding(self.G, latlon=self.latlon, laplacian_components = laplacian_components,
                                              dtype=self.dtype, device=self.device)

      # The maximum sequence length is equal to the maximum graph degree, or the
      # maximum number of neighbors a node have in the graph
      # Náo entendi isso.. pq nao podemos aumentar o sequence length?

      self.max_length = max([d for n, d in self.G.degree()]) + 1

      # precompute and store all time embeddings to save processing
      self.time_embeddings = SinusoidalTemporalEncoding(self.data['timestamp'], dtype=self.dtype, device=self.device)

      self.num_sensors = self.G.number_of_nodes()
      # print(f'PEMS dataset has {self.num_sensors} sensors')

      #self.sensors = sorted([k for k in self.G.nodes()])
      # print(f'PEMS dataset has {len(self.data)} samples')
      self.num_samples = len(self.data) - self.steps_ahead
      # print(f'PEMS dataset has {self.num_samples} samples')
      self.token_dim = 9

      self.value_index = 4

      self.tokenizer = NeighborhoodTokenizer(dtype = self.dtype, device = self.device,
                                             graph = self.G, num_nodes = self.num_sensors,
                                             max_length = self.max_length, 
                                             token_dim = self.token_dim, 
                                             value_embedder = self.value_embedder,
                                             spatial_embedding = self.node_embeddings,
                                             temporal_embedding = self.time_embeddings)
      
      self.NULL_SYMBOL = self.tokenizer.NULL_SYMBOL

      self.td = kwargs.get('use_tensordict', False)

      if self.td:
        self.to_tensordict()
        

    def to_tensordict(self):
      if not self.td:
        cols1 = self.data.columns[0]
        cols2 = self.data.columns[1:].tolist()

        df1 = self.data[[cols1]]
        df2 = self.data[cols2]

        self.data = TensorDictDataframe(dtype=self.dtype, device = self.device, 
                                        numeric_df=df2, nonnumeric_df=df1)
        self.td = True

    
    def get_sample(self, sensor, index):
      X = self.tokenizer.tokenize_sample(self.data, sensor, index)
      if not self.td:    
        y = torch.tensor(self.data[str(sensor)].values[index+self.steps_ahead], dtype=self.dtype, device=self.device)
      else:
        y = self.data[str(sensor),index+self.steps_ahead]
      return X,y

    # Will returna a SensorDataset filled with the sensor & neighbors preprocessed data (X)
    # and the expected values for t+y (y)
    def get_sensor_dataset(self, sensor, train = 0.7, dtype = torch.float64, **kwargs):
      X = self.tokenizer.tokenize_all(self.data, sensor)[:-self.steps_ahead]
      #whats the size of the X?
      print(f"Size of X for sensor {sensor}: {X.shape}")
      #can i get a sample of the X?
      print(f'Sample of X for sensor {sensor}: {X[0]}')
      y = torch.tensor(self.data[str(sensor)].values[self.steps_ahead:], dtype=self.dtype, device=self.device)
      print(f'Size of y for sensor {sensor}: {y.shape}')
      print(f'Sample of y for sensor {sensor}: {y[0]}')
      return SensorDataset(str(sensor),X,y,train, dtype, num_features = self.num_sensors,
                           max_length=self.max_length, token_dim=self.token_dim,
                           value_index=self.value_index, **kwargs)

    def get_fewsensors_dataset(self, sensors, train = 0.7, dtype = torch.float64, **kwargs):
      X = None
      y = None
      try:
        for sensor in sensors:
          tmpX = self.tokenizer.tokenize_all(self.data, sensor)[:-self.steps_ahead]
          tmpy = torch.tensor(self.data[str(sensor)].values[self.steps_ahead:], dtype=self.dtype, device=self.device)
          if X is None:
            X = tmpX
            y = tmpy
          else:
            #X = np.vstack((X,tmpX))
            X = torch.vstack((X,tmpX))
            #y = np.hstack((y,tmpy))
            y = torch.hstack((y,tmpy))
      except Exception as ex:
        print(sensor, str(ex))

      return SensorDataset('FEW',X,y,train, dtype, num_features = self.num_sensors,
                           max_length=self.max_length, token_dim=self.token_dim,
                           value_index=self.value_index, **kwargs)

    
    def get_breadth_dataset(self, start_sensor, max_sensors = 20, train = 0.7, dtype = torch.float64, **kwargs):
      sensors = []
      next = [start_sensor]
      m = 0
      while m < max_sensors:
        for sensor in next:
          if sensor not in sensors: 
            sensors.append(sensor)
            m += 1
            next.remove(sensor)
            if m < max_sensors:
              for neighbor in self.G.neighbors(sensor):
                next.append(neighbor)
            else:
              break

      return self.get_fewsensors_dataset(sensors, train = train, dtype = dtype, **kwargs), sensors

    def get_allsensors_dataset(self, **kwargs):
      return AllSensorDataset(pems=self, **kwargs)
    
    def get_sensor(self, index):
      if not self.td: 
        return int(self.data.columns[index + 1])
      else:
        return int(self.data.numeric_columns[index])

    
    def to(self, *args, **kwargs):
      if isinstance(args[0], str):
        self.device = args[0]
      else:
        self.dtype = args[0]
      return self
    
class PEMSDataset(Dataset):
    def __init__(self, data_path, edges_path, nodes_path, 
                 input_len=12, output_len=12,
                 device="cpu"):
        """Base dataset that handles all data loading"""
        # Load all data files
        self.full_data = pd.read_csv(data_path, parse_dates=['timestamp'], sep=',')
        self.edges = pd.read_csv(edges_path)
        self.nodes = pd.read_csv(nodes_path)
        
        # Process graph structure
        self.edge_index = torch.tensor(self.edges[['source', 'target']].values.T, dtype=torch.long)
        self.edge_attr = torch.tensor(self.edges['weight'].values, dtype=torch.float32)
        
        # Get sensor list
        self.sensor_nodes = sorted([int(col) for col in self.full_data.columns if col != 'timestamp'])

        self.num_nodes = len(self.sensor_nodes)
        
        # Store parameters
        self.input_len = input_len
        self.output_len = output_len
        self.device = device
        self.timestamps = sorted(self.full_data['timestamp'].unique())
        
        # Will be set during splitting
        self.split_data = None
        self.split_timestamps = None

    @classmethod
    def create_splits(cls, data_path, edges_path, nodes_path, 
                     input_len=12, output_len=12,
                     split_ratios=(0.7, 0.2, 0.1),
                     device="cpu"):
        """Factory method that returns train/val/test datasets"""
        full_dataset = cls(data_path, edges_path, nodes_path, 
                         input_len, output_len, device)
        
        # Time-based splitting indices
        timestamps = full_dataset.timestamps
        train_end = int(len(timestamps) * split_ratios[0])
        val_end = train_end + int(len(timestamps) * split_ratios[1])
        
        # Create split-specific datasets
        train_data = cls(data_path, edges_path, nodes_path,
                        input_len, output_len, device)
        train_data._set_split(timestamps[:train_end])
        
        val_data = cls(data_path, edges_path, nodes_path,
                      input_len, output_len, device)
        val_data._set_split(timestamps[train_end:val_end])
        
        test_data = cls(data_path, edges_path, nodes_path,
                       input_len, output_len, device)
        test_data._set_split(timestamps[val_end:])
        
        return train_data, val_data, test_data

    def _set_split(self, split_timestamps):
        """Internal method to set time range for this split"""
        self.split_timestamps = split_timestamps
        self.split_data = self.full_data[
            self.full_data['timestamp'].isin(split_timestamps)
        ]
        self.timestamps = sorted(self.split_data['timestamp'].unique())

    def __len__(self):
        return len(self.timestamps) - self.input_len - self.output_len + 1
        
    def __getitem__(self, idx):
        # Get input window
        input_start = idx
        input_end = input_start + self.input_len
        input_data = self.split_data.iloc[input_start:input_end]
        
        # Get output window
        output_start = input_end
        output_end = output_start + self.output_len
        output_data = self.split_data.iloc[output_start:output_end]
        
        # Prepare X (OrderedDict of timestamp -> sensor values)
        X = OrderedDict()
        for ts in input_data['timestamp']:
            ts_str = str(ts)
            X[ts_str] = OrderedDict()
            for node in self.sensor_nodes:
                X[ts_str][str(node)] = input_data.loc[input_data['timestamp'] == ts, f'{node}'].values[0]
        
        # Prepare y (tensor of future values)
        y = torch.zeros((self.output_len, self.num_nodes), dtype=torch.float32)
        for i, ts in enumerate(output_data['timestamp']):
            for j, node in enumerate(self.sensor_nodes):
                y[i,j] = output_data.loc[output_data['timestamp'] == ts, f'{node}'].values[0]
        
        return X, y.to(self.device)

    def get_graph_data(self):
        """Returns edge_index and edge_attr for the whole graph"""
        return self.edge_index.to(self.device), self.edge_attr.to(self.device)


class PEMS03(PEMSBase):
    def __init__(self,**kwargs):
      super(PEMS03, self).__init__(latlon = True, 
                                   edges_file = kwargs.pop('edges_file', "https://raw.githubusercontent.com/astorelucas/st_gnca/refs/heads/main/st_gnca/data/PEMS03/edges.csv"),
                                   nodes_file = kwargs.pop('nodes_file', "https://raw.githubusercontent.com/astorelucas/st_gnca/refs/heads/main/st_gnca/data/PEMS03/nodes.csv"),
                                   data_file = kwargs.pop('data_file', "https://raw.githubusercontent.com/astorelucas/st_gnca/refs/heads/main/st_gnca/data/PEMS03/data.csv"),
                                   **kwargs)

class PEMS04(PEMSBase):
    def __init__(self,**kwargs):
      super(PEMS04, self).__init__(latlon = False, 
                                   edges_file = kwargs.pop('edges_file', "https://raw.githubusercontent.com/petroniocandido/st_nca/refs/heads/main/st_nca/data/PEMS04/edges.csv"),
                                   data_file = kwargs.pop('data_file', "https://raw.githubusercontent.com/petroniocandido/st_nca/refs/heads/main/st_nca/data/PEMS04/data.csv"),
                                   **kwargs)

class PEMS08(PEMSBase):
    def __init__(self,**kwargs):
      super(PEMS08, self).__init__(latlon = False, 
                                   edges_file = kwargs.pop('edges_file', "https://raw.githubusercontent.com/petroniocandido/st_nca/refs/heads/main/st_nca/data/PEMS08/edges.csv"),
                                   data_file = kwargs.pop('data_file', "https://raw.githubusercontent.com/petroniocandido/st_nca/refs/heads/main/st_nca/data/PEMS08/data.csv"),
                                   **kwargs)