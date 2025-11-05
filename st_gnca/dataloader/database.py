import torch
import networkx as nx
import pandas as pd
import numpy as np

from st_gnca.embeddings.temporal import SinusoidalTemporalEncoding, MultiScaleTemporalEncoding
from st_gnca.embeddings.value import ValueEmbedding, MinMaxTransform

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32

class DataBase:
    def __init__(self, **kwargs):
        self.dtype = kwargs.get('dtype', DTYPE)
        self.device = kwargs.get('device', DEVICE)

        self.G = nx.Graph()

        self.data = pd.read_csv(kwargs.get('data_file', 'data.csv'), engine='pyarrow')
        columns_to_drop = ['']
        self.data.drop(columns=columns_to_drop, axis=1, inplace=True)

        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'].values)
        self.sensor_data_raw = self.data.drop(columns=['timestamp']).values.astype(np.float32)

        self.sensor_ids = list(self.data.columns.drop('timestamp').astype(int).values.astype(np.int32))

        self.sensor_id_map, self.reverse_sensor_id_map = self._create_sensor_id_map()

        edges = pd.read_csv(kwargs.get('edges_file', 'edges.csv'), engine='pyarrow')

        for source_id, target_id, weight in edges[['source', 'target', 'weight']].values:

            source_idx = self.sensor_id_map.get(source_id)
            target_idx = self.sensor_id_map.get(target_id)

            if source_idx is not None and target_idx is not None:
                self.G.add_edge(source_idx, target_idx, weight=weight.round(4).item())

        self.edge_index = torch.tensor(list(self.G.edges)).t().contiguous().to(self.device)
        self.edge_weight = torch.tensor([self.G[u][v]['weight'] for u, v in self.G.edges()]).to(self.device)

        self.num_sensors = self.G.number_of_nodes()
        print(f'num de sensors: {self.num_sensors}')

        self.sensor_data = self._normalize_data()

        # self.temporal_embedding = SinusoidalTemporalEncoding(
        #     emb_dim=10,
        #     dates=self.data['timestamp'],
        #     device=self.device,
        #     dtype=self.dtype
        # )

        self.temporal_embedding = MultiScaleTemporalEncoding(
            dates=self.data['timestamp'],
            emb_dim=12,
            device=self.device,
            dtype=self.dtype
        )

        self.temporal_features = self.temporal_embedding.all()

    def _create_sensor_id_map(self):
        """
        Creates a forward and reverse mapping for sensor IDs.

        Returns:
            tuple: A tuple containing the forward map (original ID -> new index)
                and the reverse map (new index -> original ID).
        """
        sorted_ids = sorted(list(set(self.sensor_ids)))
        forward_map = {original_id: new_index for new_index, original_id in enumerate(sorted_ids)}
        reverse_map = {new_index: original_id for original_id, new_index in forward_map.items()}
        return forward_map, reverse_map

    def _normalize_data(self):
        self.value_embedding = ValueEmbedding(
            self.sensor_data_raw,
            value_embedding_type='ztransform',
            dtype=self.dtype,
            device=self.device
        )
        sensor_data = self.value_embedding.forward(
            torch.tensor(self.sensor_data_raw, dtype=self.dtype, device=self.device)
        )
        return sensor_data

    def concat_features(self):
        print(f'temporal : {self.temporal_features.shape}')
        print(f'sensor_data : { self.sensor_data.shape}')
        combined = torch.cat((self.temporal_features, self.sensor_data), dim=1)
        return combined

class BatchBuilder:
    def __init__(self, data, batch_size, sequence_len, train_split=0.6, val_split=0.2, device=DEVICE, dtype=DTYPE, **kwargs):
        self.data = data
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.sequence_len = sequence_len
        self.horizon = kwargs.get('horizon', 0)

        if train_split + val_split >= 1.0:
            raise ValueError("Train + validation split must be less than 1.0")
        self.train_split = train_split
        self.val_split = val_split

        self.data_tokenized = self.data.concat_features()
        self.num_samples = self.data_tokenized.size(0)

        if self.num_samples <= self.sequence_len + self.horizon:
            raise ValueError(
                f"Dataset is too small for the given sequence length ({self.sequence_len}) and horizon ({self.horizon}). "
                f"Data size: {self.num_samples}, Required: {self.sequence_len + self.horizon}"
            )

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
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=False
        )

    def get_val_loader(self):
        dataset = SlidingWindowDataset(self.val_data, self.sequence_len, self.horizon)
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=False
        )

    def get_test_loader(self):
        dataset = SlidingWindowDataset(self.test_data, self.sequence_len, self.horizon)
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=False
        )

class SlidingWindowDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_len, horizon, device=DEVICE):
        self.data = data.to(device)
        self.seq_len = seq_len
        self.horizon = horizon
        self.device = device

        # Ensure the dataset has enough samples
        self.num_samples = data.size(0) - seq_len - horizon
        if self.num_samples <= 0:
            raise ValueError(
                f"Dataset is too small for the given sequence length ({seq_len}) and horizon ({horizon}). "
                f"Data size: {data.size(0)}, Required: {seq_len + horizon}"
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len].to(self.device)
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.horizon].to(self.device)
        return x, y