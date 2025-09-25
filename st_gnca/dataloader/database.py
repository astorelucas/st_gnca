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

        self.G = nx.Graph()

        self.data = pd.read_csv(kwargs.get('data_file','data.csv'), engine='pyarrow')
        self.data = self.data.iloc[:, :-5]
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'].values)
        self.sensor_data_raw = self.data.drop(columns=['timestamp']).values.astype(np.float32)

        self.num_sensors = self.G.number_of_nodes()

        self.sensor_ids = list(self.data.columns.drop('timestamp').astype(int).values)

        self.sensor_id_map, self.reverse_sensor_id_map = self._create_sensor_id_map()

        edges = pd.read_csv(kwargs.get('edges_file','edges.csv'), engine='pyarrow')
        for source_id, target_id, weight in edges[['source', 'target', 'weight']].values:
            # print(f"Adding edge from {source_id} to {target_id} with weight {weight}")
            # Use the mapping to get the integer indices
            source_idx = self.sensor_id_map.get(source_id)
            # print(f"Source ID: {source_id}, Mapped Index: {source_idx}")
            target_idx = self.sensor_id_map.get(target_id)
            # print(f"Target ID: {target_id}, Mapped Index: {target_idx}")

            if source_idx is not None and target_idx is not None:
                self.G.add_edge(source_idx, target_idx, weight=weight)

        self.edge_index = torch.tensor(list(self.G.edges)).t().contiguous().to(self.device)
        # print(f"Edge Index shape: {self.edge_index.shape}") # Should be [2, num_edges]

        self.edge_weight = torch.tensor([self.G[u][v]['weight'] for u,v in self.G.edges()]).to(self.device)

        self.sensor_data = self._load_data()

        self.temporal_embedding = SinusoidalTemporalEncoding(emb_dim=4, 
                                                             dates=self.data['timestamp'], 
                                                             device=self.device, 
                                                             dtype=self.dtype)

        self.temporal_features = self.temporal_embedding.all()
         

        self.max_graph_degree = max(dict(self.G.degree()).values())

    def _create_sensor_id_map(self):
        """
        Creates a forward and reverse mapping for sensor IDs.

        Args:
            sensor_ids (list or np.ndarray): A list or array of unique sensor IDs.

        Returns:
            tuple: A tuple containing the forward map (original ID -> new index)
                and the reverse map (new index -> original ID).
        """
        # 1. Sort the unique sensor IDs to ensure a consistent mapping
        sorted_ids = sorted(list(set(self.sensor_ids)))

        # 2. Create the forward map (original ID -> new index)
        forward_map = {original_id: new_index for new_index, original_id in enumerate(sorted_ids)}

        # 3. Create the reverse map (new index -> original ID)
        reverse_map = {new_index: original_id for original_id, new_index in forward_map.items()}

        return forward_map, reverse_map

    def _load_data(self):
        # Extract sensor data and convert to tensor

        self.value_embedding = ValueEmbedding(self.sensor_data_raw, 
                                               value_embedding_type='minmax',
                                               dtype=self.dtype, 
                                               device=self.device)
        
        sensor_data = self.value_embedding.forward(torch.tensor(self.sensor_data_raw, dtype=self.dtype, device=self.device))
        # print(f"Sensor data shape: {sensor_data.shape}") # (26208, 64)
        # print(f"Sensor data sample: {sensor_data[:10, :10]}")
        return sensor_data

    def concat_features(self):
        # Concatenate temporal embeddings with all sensor features for all timestamps
        combined = torch.cat((self.temporal_features, self.sensor_data), dim=1)
        # print(f"Combined features shape: {combined.shape}") # ([26208, 358, 5])
        return combined

class BatchBuilder:
    def __init__(self, data, batch_size, sequence_len, train_split=0.7, val_split=0.1, device=DEVICE, dtype=DTYPE, **kwargs):
        self.data_tokenized = data.concat_features()
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.num_samples = data.sensor_data.size(0)
        self.num_sensors = data.num_sensors
        self.sequence_len = sequence_len
        self.train_split = train_split
        self.val_split = val_split
        self.horizon = kwargs.get('horizon', 0)

        # Validate splits
        assert self.train_split + self.val_split < 1.0, "Train + validation split must be less than 1.0"

        self.train_data, self.val_data, self.test_data = self._split_data()

    def _split_data(self):
        train_end = int(self.num_samples * self.train_split)
        val_end = int(self.num_samples * (self.train_split + self.val_split))

        train_data = self.data_tokenized[:train_end]
        val_data = self.data_tokenized[train_end:val_end]
        test_data = self.data_tokenized[val_end:]

        return train_data, val_data, test_data

    def get_train_loader(self):
        dataset = SlidingWindowDataset(self.train_data, self.sequence_len, self.horizon)
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )

    def get_val_loader(self):
        dataset = SlidingWindowDataset(self.val_data, self.sequence_len, self.horizon)
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

    def get_test_loader(self):
        dataset = SlidingWindowDataset(self.test_data, self.sequence_len, self.horizon)
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

class SlidingWindowDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_len, horizon):
        self.data = data
        self.seq_len = seq_len
        self.horizon = horizon
        self.num_samples = data.size(0) - seq_len - horizon

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.horizon]
        return x, y