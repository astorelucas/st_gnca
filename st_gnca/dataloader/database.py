import torch
import networkx as nx
import pandas as pd
import numpy as np

from torch_geometric.data import Data
from st_gnca.embeddings.temporal import SinusoidalTemporalEncoding
from st_gnca.embeddings.value import ValueEmbedding

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32

class DataBase:
    def __init__(self, **kwargs):
        self.dtype = kwargs.get('dtype', DTYPE)
        self.device = kwargs.get('device', DEVICE)

        edges = pd.read_csv(kwargs.get('edges_file','edges.csv'), engine='pyarrow')

        # Create the graph
        self.G = nx.Graph()
        for row in edges.iterrows():
            self.G.add_edge(int(row[1]['source']),int(row[1]['target']), weight=row[1]['weight'])

        del(edges)

        self.data = pd.read_csv(kwargs.get('data_file','data.csv'), engine='pyarrow')
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'].values)

        self.num_sensors = self.G.number_of_nodes()

        self.sensor_ids = list(self.G.nodes())

        self.edge_index = torch.tensor(list(self.G.edges)).t().contiguous().to(self.device)

        self.edge_weight = torch.tensor([self.G[u][v]['weight'] for u,v in self.G.edges()]).to(self.device)

        self.num_edges = self.edge_index.size(1)

        self.sensor_data, self.adj_matrix = self._load_data()

        self.temporal_embedding = SinusoidalTemporalEncoding(emb_dim=4, 
                                                             dates=self.data['timestamp'], 
                                                             device=self.device, 
                                                             dtype=self.dtype)

        self.temporal_features = self.temporal_embedding.all()

        self.max_graph_degree = max(dict(self.G.degree()).values())

    def _load_data(self):
        # Extract sensor data and convert to tensor
        sensor_data = self.data.drop(columns=['timestamp']).values  # Shape: [n_timesteps, n_sensors]

        self.value_embedding = ValueEmbedding(sensor_data, 
                                               value_embedding_type='scaling',
                                               dtype=self.dtype, 
                                               device=self.device)
        sensor_data = self.value_embedding(torch.tensor(sensor_data, dtype=self.dtype, device=self.device))

        # Create adjacency matrix
        adj_matrix = nx.to_numpy_array(self.G, nodelist=self.sensor_ids)
        adj_matrix = torch.tensor(adj_matrix, dtype=self.dtype, device=self.device)

        return sensor_data, adj_matrix
        
    def get_adj_matrix(self):
        return self.adj_matrix

    def concat_features(self):
        # Concatenate temporal embeddings with all sensor features for all timestamps
        combined = torch.cat((self.temporal_features, self.sensor_data), dim=1)
        return combined

class BatchBuilder:
    def __init__(self, data, batch_size, sequence_len, train_split=0.7, val_split=0.1,device=DEVICE, dtype=DTYPE):
        self.data_tokenized = data.concat_features()
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.num_samples = data.sensor_data.size(0)
        self.num_sensors = data.num_sensors
        self.sequence_len = sequence_len
        self.train_split = train_split
        self.val_split = val_split

        self.train_data, self.val_data, self.test_data = self._split_data()


    def _split_data(self):

        assert self.train_split + self.val_split < 1.0

        train_end = int(self.num_samples * self.train_split)
        val_end = int(self.num_samples * (self.train_split + self.val_split))

        train_data = self.data_tokenized[:train_end]
        val_data = self.data_tokenized[train_end:val_end]
        test_data = self.data_tokenized[val_end:]

        return train_data, val_data, test_data
    
    def _create_sequences(self, data):
        X, Y = [], []
        num_timesteps = data.size(0)

        for i in range(num_timesteps - self.sequence_len):
            input_window = data[i:i+self.sequence_len]
            target_window = data[i+self.sequence_len]
            X.append(input_window)
            Y.append(target_window)

        # stack list of tensors into a single tensor: shapes -> X: [number_of_sequences, seq_len, num_sensors], Y: [number_of_sequences, num_sensors]
        X = torch.stack(X) if len(X) > 0 else torch.empty((0,), dtype=self.dtype, device=self.device)
        Y = torch.stack(Y) if len(Y) > 0 else torch.empty((0,), dtype=self.dtype, device=self.device)
        return X.to(dtype=self.dtype, device=self.device), Y.to(dtype=self.dtype, device=self.device)
    
    def get_train_loader(self):
        X, Y = self._create_sequences(self.train_data)
        dataset = torch.utils.data.TensorDataset(X, Y)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    
    def get_val_loader(self):
        X, Y = self._create_sequences(self.val_data)
        dataset = torch.utils.data.TensorDataset(X, Y)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
    
    def get_test_loader(self):
        X, Y = self._create_sequences(self.test_data)
        dataset = torch.utils.data.TensorDataset(X, Y)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)