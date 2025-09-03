import torch
import networkx as nx
import pandas as pd
import numpy as np


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32
DEFAULT_PATH = 'st_gnca/'
DATA_PATH = DEFAULT_PATH + 'data/PEMS03/'


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

        def 



def build_data():

    # 1. Entrar com o dataset
    data_path = DATA_PATH + "data.csv"
    edges_path = DATA_PATH + "edges.csv"
    nodes_path = DATA_PATH + "nodes.csv"

    print("Loading historical sensor data...")
    df_data = pd.read_csv(data_path, index_col=0)  # Assuming first col is timestamp
    # Calculate the mean flow for each sensor (each column) across all timestamps
    # This gives us one feature per node.
    node_feature_matrix = df_data.values  # Shape: [n_timesteps, n_nodes]
    node_features = torch.tensor(node_feature_matrix.mean(axis=0), dtype=torch.float).view(-1, 1)  # Shape: [n_nodes, 1]

    print(f"Historical data shape: {node_feature_matrix.shape}")
    print(f"Node features shape: {node_features.shape}")

    # Load graph structure (edges)
    print("Loading graph edges...")
    df_edges = pd.read_csv(edges_path)
    # PyG expects edge_index in the format [2, num_edges]
    edge_index = torch.tensor([df_edges['source'].values, df_edges['target'].values], dtype=torch.long)
    # If you have weights, you can add them as edge attributes
    edge_weight = torch.tensor(df_edges['weight'].values, dtype=torch.float)

    print(f"Number of edges: {edge_index.shape[1]}")

    # Load node positions (optional, for later analysis/visualization)
    print("Loading node metadata...")
    df_nodes = pd.read_csv(nodes_path)
    node_positions = torch.tensor(df_nodes[['lat', 'long']].values, dtype=torch.float)

    # Get the number of nodes from the feature data
    num_nodes = node_features.size(0)

    # ------------------------------
    # Create the PyG Data Object
    # ------------------------------
    data = Data(
        x=node_features,           # Node features [num_nodes, num_features]
        edge_index=edge_index,     # Graph connectivity [2, num_edges]
        edge_attr=edge_weight,     # Edge weights [num_edges, ]
        pos=node_positions,        # Node coordinates [num_nodes, 2]
        num_nodes=num_nodes        # Number of nodes in the graph
    )

    # (Optional) Normalize node features
    data.x = (data.x - data.x.mean()) / data.x.std()

    print("\nFinal Data Object Summary:")
    print("==========================")
    print(data)
    print(f"\nNumber of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of node features: {data.num_features}")
    print(f"Has edge weights: {data.edge_attr is not None}")
    print(f"Has node positions: {data.pos is not None}")

    return data
