import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import networkx as nx
from typing import Optional, Union

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SpatialEmbedding(nn.Module):
    """
    A module that computes Laplacian Eigenvector Positional Encodings for all nodes in a graph.
    
    This class handles both dense and sparse computation methods and provides options for
    different normalization techniques.
    
    Args:
        graph: Input graph. Can be a NetworkX graph, an adjacency matrix (numpy array or scipy sparse),
               or an edge list.
        k (int): Number of Laplacian eigenvectors to use (dimensionality of the embedding).
        normalization (str): Type of Laplacian normalization. Options: 'sym' (symmetric), 'rw' (random walk).
        use_eigsh (bool): Whether to use sparse eigenvalue decomposition (recommended for large graphs).
        which (str): Which eigenvalues to compute ('SM' for smallest magnitude, 'LM' for largest).
        skip_first (bool): Whether to skip the first trivial eigenvector (constant component).
        **kwargs: Additional arguments to pass to the eigenvalue solver.
    """
    
    def __init__(self, graph, k: int = 8, normalization: str = 'sym', 
                 use_eigsh: bool = True, which: str = 'SM', skip_first: bool = True,
                 **kwargs):
        super().__init__()
        
        self.k = k
        self.normalization = normalization
        self.use_eigsh = use_eigsh
        self.which = which
        self.skip_first = skip_first
        self.kwargs = kwargs
        
        # Convert input graph to appropriate format and compute Laplacian
        laplacian = self._process_graph(graph)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = self._compute_eigen_decomposition(laplacian)
        
        # Create the positional encoding matrix
        self.pos_enc = self._create_positional_encoding(eigenvalues, eigenvectors)
        
        # Register as buffer so it moves with the model and is not a parameter
        self.register_buffer('embedding', self.pos_enc)
    
    def _process_graph(self, graph):
        """Convert input graph to appropriate Laplacian matrix."""
        if isinstance(graph, nx.Graph):
            # Convert NetworkX graph to Laplacian
            if self.normalization == 'sym':
                L = nx.normalized_laplacian_matrix(graph)
            elif self.normalization == 'rw':
                L = nx.directed_laplacian_matrix(graph)  # For random walk normalization
            else:
                L = nx.laplacian_matrix(graph)  # Unnormalized Laplacian
            return L
        
        elif isinstance(graph, (np.ndarray, csr_matrix)):
            # Assume input is adjacency matrix
            A = graph
            if isinstance(A, np.ndarray):
                A = csr_matrix(A)
            
            # Compute degree matrix
            degrees = np.array(A.sum(axis=1)).flatten()
            D = csr_matrix((degrees, (np.arange(len(degrees)), np.arange(len(degrees)))))
            
            # Compute appropriate Laplacian
            if self.normalization == 'sym':
                D_inv_sqrt = csr_matrix((1.0 / np.sqrt(degrees)), 
                                       (np.arange(len(degrees)), np.arange(len(degrees))))
                L = csr_matrix.eye(len(degrees)) - D_inv_sqrt @ A @ D_inv_sqrt
            else:
                L = D - A  # Unnormalized Laplacian
            
            return L
        
        else:
            raise ValueError("Unsupported graph format. Use NetworkX graph, numpy array, or scipy sparse matrix.")
    
    def _compute_eigen_decomposition(self, laplacian):
        """Compute eigenvalue decomposition using appropriate method."""
        n_nodes = laplacian.shape[0]
        
        if self.use_eigsh and n_nodes > 100:  # Use sparse method for larger graphs
            # Calculate how many eigenvectors to request
            k_request = self.k + 1 if self.skip_first else self.k
            
            # Ensure we don't request more eigenvectors than nodes
            k_request = min(k_request, n_nodes - 1)
            
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = eigsh(
                laplacian, 
                k=k_request, 
                which=self.which,
                **self.kwargs
            )
        else:
            # Convert to dense matrix and use numpy
            if hasattr(laplacian, 'toarray'):
                laplacian = laplacian.toarray()
            
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        
        return eigenvalues, eigenvectors
    
    def _create_positional_encoding(self, eigenvalues, eigenvectors):
        """Create the positional encoding matrix from eigenvalues and eigenvectors."""
        # Sort by eigenvalues in ascending order
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Determine which eigenvectors to use
        start_idx = 1 if self.skip_first else 0
        end_idx = start_idx + self.k
        
        # Ensure we don't exceed available eigenvectors
        end_idx = min(end_idx, eigenvectors.shape[1])
        
        # Select the appropriate eigenvectors
        pos_enc = eigenvectors[:, start_idx:end_idx]
        
        # Handle sign ambiguity - you can also make this learnable
        # Here we use absolute value as one common approach
        pos_enc = np.abs(pos_enc)
        
        # Convert to torch tensor
        return torch.tensor(pos_enc, dtype=torch.float32)
    
    def forward(self, node_indices: Optional[torch.Tensor] = None, 
                batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass returns positional embeddings.
        
        Args:
            node_indices: Optional tensor of node indices to return embeddings for.
            batch_size: Optional batch size for memory-efficient retrieval.
            
        Returns:
            torch.Tensor: Positional embeddings of shape [n_nodes, k] or [len(node_indices), k]
        """
        if node_indices is None:
            return self.embedding
        
        if batch_size is None:
            return self.embedding[node_indices]
        else:
            # Memory-efficient batch retrieval
            results = []
            for i in range(0, len(node_indices), batch_size):
                batch_indices = node_indices[i:i + batch_size]
                results.append(self.embedding[batch_indices])
            return torch.cat(results, dim=0)
    
    def extra_repr(self) -> str:
        """Extra representation string for printing."""
        str_repr = f'k={self.k}, normalization={self.normalization}, use_eigsh={self.use_eigsh}, which={self.which}, skip_first={self.skip_first}'
        return str_repr