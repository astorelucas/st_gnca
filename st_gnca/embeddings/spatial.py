import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import networkx as nx
from typing import Optional, Union
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import numpy as np
import networkx as nx

from sklearn.manifold import SpectralEmbedding

import torch
from torch import nn

from tensordict import TensorDict


class SpatialEmbedding(nn.Module):
  def __init__(self, graph, laplacian_components = 20, **kwargs):
    super().__init__()
    self.device = kwargs.get('device','cpu')
    self.dtype = kwargs.get('dtype',torch.float32)
    tmp_dict = {}
    self.laplacian_components = laplacian_components
    self.graph = graph

    M = nx.adjacency_matrix(graph).todense()
    laplacian = SpectralEmbedding(n_components=laplacian_components) #, affinity='precomputed')
    laplacian_map = laplacian.fit_transform(M)

    self.length = 0
    for node in sorted(self.graph.nodes()):
        emb = np.zeros(self.laplacian_components)
        emb = laplacian_map[node,:]
        tmp_dict[str(node)] = torch.tensor(emb, dtype = self.dtype, device = self.device)
        self.length += 1

    self.embeddings = TensorDict(tmp_dict)

  def forward(self, node):
     return self.embeddings[str(node)]

  def __getitem__(self,  node):
     return self.embeddings[str(node)]

  def all(self):
    ret = torch.empty(self.length, self.laplacian_components,
                        dtype=self.dtype, device=self.device)
    for it,emb in enumerate(self.embeddings.values(sort=True)):
      ret[it, :] = emb
    return ret

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]
    self.embeddings = self.embeddings.to(*args, **kwargs)

    return self