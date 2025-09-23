import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_scaler(train_ds, device="cuda", sample_size=100_000):
    """Optimized scaler initialization with random sampling"""
    # 1. Estimate min/max from a random sample (no full dataset iteration)
    sample_indices = np.random.choice(len(train_ds), size=min(sample_size, len(train_ds)), replace=False)
    
    # 2. Vectorized extraction using dataset.__getitem__
    sample_ys = []
    for idx in sample_indices:
        _, y = train_ds[idx]  # Direct access avoids DataLoader overhead
        sample_ys.append(y.cpu().numpy() if torch.is_tensor(y) else y)
    
    # 3. Stack and compute stats in one batch
    scaler = ScalingTransform(np.stack(sample_ys), device=device)
    
    print(f"Scaler initialized with {len(sample_ys)} samples (min={scaler.min:.4f}, max={scaler.max:.4f})")
    return scaler

class ValueEmbedding(nn.Module):
  def __init__(self, data, **kwargs):
    super().__init__()
    self.device = kwargs.get('device',DEVICE)
    self.dtype = kwargs.get('dtype',torch.float32)
    self.value_embedding_type = kwargs.get('value_embedding_type','normalization')

    if self.value_embedding_type == 'normalization':
      self.embedder = ZTransform(data, **kwargs)
    elif self.value_embedding_type == 'scaling':
      self.embedder = ScalingTransform(data, **kwargs)
    elif self.value_embedding_type == 'minmax':
      self.embedder = MinMaxTransform(data, **kwargs)
    else:
      raise ValueError("Unknown embedder type!")
    
  def forward(self, x):
     return self.embedder.forward(x)
  
  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]
    self.embedder = self.embedder.to(*args, **kwargs)

    return self

class MinMaxTransform(nn.Module):
    def __init__(self, data, **kwargs):
        super().__init__()
        self.min_val = data.min()
        self.max_val = data.max()
        self.range = self.max_val - self.min_val

    def forward(self, x):
        """
        Performs Min-Max normalization.
        """
        # Add a small epsilon to prevent division by zero in case of a zero range.
        return (x - self.min_val) / (self.range + 1e-8)

    def denormalize(self, x_normalized):
        """
        Reverts the Min-Max normalization.
        """
        return x_normalized * self.range + self.min_val
    
class ScalingTransform(nn.Module):
    def __init__(self, data, **kwargs):
        super().__init__()
        self.device = kwargs.get('device', 'cpu')
        self.dtype = kwargs.get('dtype', torch.float32)
        self.epsilon = kwargs.get('epsilon', 1e-8)
        
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=self.dtype, device=self.device)
        
        # Safely handle NaN/Inf
        data = torch.nan_to_num(
            data, 
            nan=0.0, 
            posinf=torch.finfo(self.dtype).max, 
            neginf=torch.finfo(self.dtype).min
        )
        
        self.register_buffer('min', torch.min(data))
        self.register_buffer('range', (torch.max(data) - self.min) + self.epsilon)

    def forward(self, x):
        return (x - self.min) / self.range

    def denormalize(self, x):
        return x * self.range + self.min

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        if len(args) > 0:
            if isinstance(args[0], torch.dtype):
                self.dtype = args[0]
            elif isinstance(args[0], (str, torch.device)):
                self.device = args[0] if isinstance(args[0], str) else args[0].type
        return self
  
class ZTransform(nn.Module):
  def __init__(self, data, **kwargs):
    super().__init__()
    self.device = kwargs.get('device','cpu')
    self.dtype = kwargs.get('dtype',torch.float32)
    if not isinstance(data, torch.Tensor):
      data = torch.tensor(data, dtype=self.dtype, device=self.device)
    self.mu = torch.nanmean(data)
    self.sigma = torch.std(torch.nan_to_num(data,0,0,0))
    
  def forward(self, x):
     #return z(x, self.mu, self.sigma)
     return (x - self.mu) / self.sigma
  
  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]
    self.mu = self.mu.to(*args, **kwargs)
    self.sigma = self.sigma.to(*args, **kwargs)

    return self
  
class LearnableValueEmbedding(nn.Module):
    """
    A flexible module to handle and embed various types of node features.
    It combines numerical and categorical feature processing.
    """
    def __init__(self, num_nodes, numerical_dim=0, categorical_dims=None, embedding_dim=128, init_embedding=None):
        """
        Args:
            num_nodes (int): Number of nodes in the graph.
            numerical_dim (int): Dimensionality of the numerical features.
            categorical_dims (list of int): A list of the number of unique categories
                                            for each categorical feature.
            embedding_dim (int): The output dimensionality for the final node features.
            init_embedding (torch.Tensor, optional): A pre-computed tensor of node features.
                                                     If provided, it will be used as a static embedding.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.numerical_dim = numerical_dim
        self.categorical_dims = categorical_dims if categorical_dims is not None else []

        # Option 1: Use a pre-computed feature matrix (e.g., from BERT, PCA, etc.)
        if init_embedding is not None:
            self.feature_projection = nn.Linear(init_embedding.size(1), embedding_dim)
            self.init_embedding = nn.Parameter(init_embedding, requires_grad=False) # don't train the initial values
            self.mode = 'precomputed'

        # Option 2: Learnable "Input" Embedding (like in a Language Model)
        # Use this if nodes have no features, only IDs.
        elif numerical_dim == 0 and not self.categorical_dims:
            self.node_embedding = nn.Embedding(num_nodes, embedding_dim)
            self.mode = 'id'

        # Option 3: Learn from raw numerical and categorical features
        else:
            self.mode = 'raw'
            # Embedding layers for categorical features
            self.cat_embeddings = nn.ModuleList()
            for num_categories in self.categorical_dims:
                # Rule of thumb: embedding dim = min(50, round(sqrt(num_categories)) + 1)
                emb_dim = min(50, (num_categories // 2) + 1)
                self.cat_embeddings.append(nn.Embedding(num_categories, emb_dim))

            # Calculate total dimension after processing all features
            total_embedded_dim = numerical_dim
            for emb_layer in self.cat_embeddings:
                total_embedded_dim += emb_layer.embedding_dim

            # Project the concatenated features to the desired embedding dimension
            self.feature_projection = nn.Linear(total_embedded_dim, embedding_dim)

    def forward(self, node_indices=None):
        """
        Args:
            node_indices (torch.Tensor): Indices of nodes to get features for.
                                         If None, returns features for all nodes.
        Returns:
            torch.Tensor: Node feature matrix of shape [n_nodes, embedding_dim]
        """
        if self.mode == 'precomputed':
            x = self.init_embedding
            if node_indices is not None:
                x = x[node_indices]
            return self.feature_projection(x)

        elif self.mode == 'id':
            if node_indices is None:
                node_indices = torch.arange(self.node_embedding.num_embeddings, device=self.node_embedding.weight.device)
            return self.node_embedding(node_indices)

        elif self.mode == 'raw':
            # In a real scenario, you would pass numerical and categorical data here.
            # This is a placeholder. You would need to load/store these features elsewhere.
            # For example: x_num = self.numerical_features[node_indices]
            #              x_cat_list = [cat_feat[node_indices] for cat_feat in self.categorical_features]
            raise NotImplementedError("Forward pass for raw features requires stored feature data. See example below.")

    def extra_repr(self):
        return f'mode={self.mode}, embedding_dim={self.embedding_dim}'

