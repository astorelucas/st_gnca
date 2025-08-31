import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
import os
from datetime import datetime

from st_gnca.embeddings.value import ValueEmbedding
from st_gnca.embeddings.spatial import SpatialEmbedding
from st_gnca.embeddings.temporal import SinusoidalTemporalEncoding

class GraphTransformer(nn.Module):
    def __init__(self, num_nodes: int, input_dim: int, embedding_dim: int, 
                 dates: List[datetime], k: int = 10, device: torch.device = None):

        super().__init__()
        
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.k = k
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Value embedding (projection linear layer)
        self.value_embedding = nn.Linear(input_dim, embedding_dim)
        
        # Node embedding components
        self.node_embedding_proj = nn.Linear(k, embedding_dim)

        self.temporal_embedder = SinusoidalTemporalEncoding(
            dates, d_model=embedding_dim, device=self.device
        )

        # Fusion layer to combine all embeddings
        # Modificar esse fusion (talvez).. tentar somar mesmo ou aumentar o 
        self.fusion_layer = nn.Linear(embedding_dim * 3, embedding_dim)
        
        # Initialize laplacian matrix (will be computed once)
        self.register_buffer('laplacian_eigenvectors', None)
        self.register_buffer('laplacian_eigenvalues', None)
        
        # Store the graph data for sample retrieval
        self.data_tensor = None
        self.node_mapping = None
        self.timestamp_mapping = None
        self.timestamps = None
        
    def load_data(self, data_path: str, edges_path: str, nodes_path: str) -> Data:
        """
        Args:
            data_path: Path to data.csv
            edges_path: Path to edges.csv
            nodes_path: Path to nodes.csv
            
        Returns:
            PyG Data object with the graph structure
        """
        # Load data.csv
        print("Loading traffic data...")
        data_df = pd.read_csv(data_path)
        
        # Parse timestamps and create mapping
        print("Processing timestamps...")
        self.timestamps = data_df.iloc[:, 0].values
        self.timestamp_mapping = {pd.Timestamp(ts): idx for idx, ts in enumerate(self.timestamps)}
        
        # Extract sensor data
        sensor_data = data_df.iloc[:, 1:].values.astype(np.float32)  # [num_timesteps, num_nodes]
        
        # Load nodes.csv to get sensor mapping
        print("Loading node information...")
        nodes_df = pd.read_csv(nodes_path)
        sensor_ids = nodes_df['sensor'].values
        self.node_mapping = {sensor_id: idx for idx, sensor_id in enumerate(sensor_ids)}
        self.reverse_node_mapping = {idx: sensor_id for sensor_id, idx in self.node_mapping.items()}
        
        # Verify that data columns match node mapping
        if sensor_data.shape[1] != len(self.node_mapping):
            print(f"Warning: Data has {sensor_data.shape[1]} sensors but node mapping has {len(self.node_mapping)} sensors")
            # Use minimum of both
            num_available_sensors = min(sensor_data.shape[1], len(self.node_mapping))
            sensor_data = sensor_data[:, :num_available_sensors]
        
        # Load edges.csv to build graph
        print("Loading edge information...")
        edges_df = pd.read_csv(edges_path)
        
        # Create edge_index and edge_weight
        edge_indices = []
        edge_weights = []
        
        for _, row in edges_df.iterrows():
            source = self.node_mapping.get(row['source'], -1)
            target = self.node_mapping.get(row['target'], -1)
            
            if source != -1 and target != -1:
                edge_indices.append([source, target])
                edge_weights.append(row['weight'])
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous().to(self.device)
        edge_weight = torch.tensor(edge_weights, dtype=torch.float).to(self.device)
        
        # Reshape sensor data for PyG: [num_nodes, num_timesteps, input_dim]
        sensor_data = sensor_data.T  # [num_nodes, num_timesteps]
        sensor_data = sensor_data[..., np.newaxis]  # [num_nodes, num_timesteps, 1]
        
        # Store data tensor for sample retrieval
        self.data_tensor = torch.FloatTensor(sensor_data).to(self.device)  # [num_nodes, T, input_dim]
        
        # Compute Laplacian embedding
        print("Computing Laplacian embedding...")
        self.laplacian_eigenvectors, self.laplacian_eigenvalues = self.compute_laplacian_embedding(edge_index, edge_weight)
        
        # Create PyG data object
        graph_data = Data(
            x=torch.FloatTensor(sensor_data).to(self.device),  # [num_nodes, T, input_dim]
            edge_index=edge_index,
            edge_attr=edge_weight,
            num_nodes=self.num_nodes
        )

        print(f"Graph built with {self.num_nodes} nodes, {edge_index.shape[1]} edges, {len(self.timestamps)} timestamps")
        return graph_data

    def compute_laplacian_embedding(self, edge_index: torch.Tensor, 
                                   edge_weight: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Laplacian eigenvectors for node embedding using sparse operations
        """
        try:
            # Get normalized Laplacian
            lap_index, lap_weight = get_laplacian(edge_index, edge_weight, normalization='sym')
            
            # Convert to sparse matrix for eigenvalue computation
            from scipy.sparse import coo_matrix, linalg
            import numpy as np
            
            # Convert to scipy sparse matrix
            lap_index_np = lap_index.cpu().numpy()
            lap_weight_np = lap_weight.cpu().numpy()
            
            L_sparse = coo_matrix((lap_weight_np, (lap_index_np[0], lap_index_np[1])), 
                                 shape=(self.num_nodes, self.num_nodes))
            
            # Compute k smallest eigenvalues and eigenvectors
            eigenvalues, eigenvectors = linalg.eigsh(L_sparse, k=min(self.k+1, self.num_nodes-1), which='SM')
            
            # Use k smallest non-zero eigenvectors (skip the first zero eigenvalue)
            eigenvectors = eigenvectors[:, 1:self.k+1]  # Shape: [num_nodes, k]
            eigenvalues = eigenvalues[1:self.k+1]       # Shape: [k]
            
            eigenvectors = torch.FloatTensor(eigenvectors).to(self.device)
            eigenvalues = torch.FloatTensor(eigenvalues).to(self.device)
            
            return eigenvectors, eigenvalues
            
        except Exception as e:
            print(f"Error computing Laplacian embedding: {e}")
            print("Using random initialization as fallback")
            
            # Fallback: random eigenvectors
            eigenvectors = torch.randn(self.num_nodes, self.k).to(self.device)
            eigenvalues = torch.ones(self.k).to(self.device)
            
            return eigenvectors, eigenvalues
    
    def forward(self, x: Optional[torch.Tensor] = None, t: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Forward pass for all nodes at timestamp t
            
            Returns:
                value_emb: Value embeddings [batch_size, num_nodes, embedding_dim]
                node_emb: Node embeddings [batch_size, num_nodes, embedding_dim]
                temporal_emb: Temporal embeddings [batch_size, embedding_dim]
            """
            if x is None:
                if self.data_tensor is None:
                    raise ValueError("No data available. Call load_pems03_data first.")
                if t is None:
                    raise ValueError("Timestamp t must be specified when using stored data.")
                
                x = self.data_tensor[:, t, :]  # [num_nodes, input_dim]
                x = x.unsqueeze(0)  # Add batch dimension [1, num_nodes, input_dim]
            
            if len(x.shape) == 2:
                x = x.unsqueeze(0)  # Add batch dimension
            
            batch_size = x.size(0)
            
            # Value embedding
            value_emb = self.value_embedding(x)  # [batch_size, num_nodes, embedding_dim]
            
            # Node embedding from Laplacian
            if self.laplacian_eigenvectors is not None:
                node_emb = self.node_embedding_proj(self.laplacian_eigenvectors)  # [num_nodes, embedding_dim]
                node_emb = node_emb.unsqueeze(0).expand(batch_size, -1, -1)  # Add batch dimension
            else:
                node_emb = torch.randn(batch_size, self.num_nodes, self.embedding_dim).to(self.device)
            
            # Temporal embedding
            if t is not None:
                t_tensor = torch.tensor(t, device=self.device).repeat(batch_size)
                temporal_emb = self.temporal_embedder(t_tensor)  # [batch_size, embedding_dim]
                temporal_emb = temporal_emb.unsqueeze(1).expand(-1, self.num_nodes, -1)  # [batch_size, num_nodes, embedding_dim]
            else:
                temporal_emb = torch.zeros(batch_size, self.num_nodes, self.embedding_dim).to(self.device)
            
            return value_emb, node_emb, temporal_emb
    
    def get_sample(self, node_index: int, timestamp: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all three embeddings for a specific node and timestamp
        """
        if self.data_tensor is None:
            raise ValueError("Load data first using load_pems03_data()")
        
        # Convert to internal indices
        node_idx = self.node_mapping[node_index]
        
        # Convert to pandas Timestamp and then to string in the same format
        ts_query = pd.Timestamp(timestamp)

        t_idx = self.timestamp_mapping.get(ts_query)
        if t_idx is None:
            raise KeyError(f"Timestamp {ts_query} not found in mapping.")

        # Get value embedding
        x_t = self.data_tensor[node_idx, t_idx, :]  # [input_dim]
        value_emb = self.value_embedding(x_t)  # [embedding_dim]
        
        # Get node embedding
        if self.laplacian_eigenvectors is not None:
            node_emb = self.node_embedding_proj(self.laplacian_eigenvectors[node_idx])  # [embedding_dim]
        else:
            node_emb = torch.randn(self.embedding_dim).to(self.device)
        
        # Get temporal embedding (using your improved class)
        temporal_emb = self.temporal_embedder(timestamp)  # [embedding_dim]
        
        return value_emb, node_emb, temporal_emb
    
    def get_combined_embedding(self, node_index: int, timestamp: str) -> torch.Tensor:
        """
        Get fused embedding combining all three components
        """
        value_emb, node_emb, temporal_emb = self.get_sample(node_index, timestamp)
        
        combined = torch.cat([value_emb, node_emb, temporal_emb], dim=0)
        return self.fusion_layer(combined)  # [embedding_dim]
