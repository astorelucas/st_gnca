import numpy as np
import networkx as nx
from sklearn.manifold import SpectralEmbedding
import torch
from torch import nn
from tensordict import TensorDict

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SpatialEmbedding(nn.Module):
  def __init__(self, graph, laplacian_components = 2, **kwargs):
    super().__init__()
    self.device = kwargs.get('device',DEVICE)
    self.dtype = kwargs.get('dtype',torch.float32)
    tmp_dict = {}
    self.latlon = kwargs.get("latlon",True)
    if self.latlon:
      lat_dict = nx.get_node_attributes(graph,'lat')
      lon_dict = nx.get_node_attributes(graph,'lon')

      lat = [v for v in lat_dict.values()]
      lon = [v for v in lon_dict.values()]
      lat_min, lat_max = np.min(lat), np.max(lat)
      lat_rng = lat_max-lat_min
      lon_min, lon_max = np.min(lon), np.max(lon)
      lon_rng = lon_max-lon_min

    M = nx.adjacency_matrix(graph).todense()
    laplacian = SpectralEmbedding(n_components=laplacian_components, affinity='precomputed') 
    laplacian_map = laplacian.fit_transform(M)

    self.length = 0
    for ix, node in enumerate(graph.nodes()):
        emb = np.zeros(4)
        if self.latlon:
          emb[0] = (lat_dict[node] - lat_min) / lat_rng * 2 - 1
          emb[1] = (lon_dict[node] - lon_min) / lon_rng * 2 - 1
          emb[2:] = laplacian_map[ix,:]
        else:
          emb = laplacian_map[ix,:]
        tmp_dict[str(node)] = torch.tensor(emb, dtype = self.dtype, device = self.device)
        self.length += 1

    self.embeddings = TensorDict(tmp_dict) 
    
  def forward(self, node):
     return self.embeddings[str(node)]
  
  def __getitem__(self,  node):
     return self.embeddings[str(node)]
  
  def all(self):
    ret = torch.empty(self.length, 4,
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