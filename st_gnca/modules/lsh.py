#migrated and adpated from: https://github.com/petroniocandido/clshq_tk/blob/main/clshq_tk/modules/lsh.py

import numpy as np
import pandas as pd

import torch
from torch import nn

def gram_schmidt(A):
  # From: https://zerobone.net/blog/cs/gram-schmidt-orthogonalization/
  n, m = A.size()
  for i in range(m):        
    q = A[:, i] # i-th column of A
        
    for j in range(i):
      q = q - torch.dot(A[:, j], A[:, i]) * A[:, j]
        
    if np.array_equal(q, np.zeros(q.shape)):
      raise Exception("The column vectors are not linearly independent")
        
    # normalize q
    q = q / torch.sqrt(torch.dot(q, q))
        
    # write the vector back in the matrix
    A[:, i] = q
  return A


def step(x):
  return torch.heaviside(x, torch.tensor([0], dtype=x.dtype, device=x.device))

# Functions for vectorize the LSH hashing generation over the batches using torch.vmap

def f_instance_level_map(x,y):
  return torch.sum(x * y)

instance_level_map = torch.func.vmap(f_instance_level_map, in_dims=0)

def f_batch_level_map(x, embed_dim, weigths):
  return instance_level_map(x.repeat(embed_dim,1,1), weigths)


class LSH(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()
    output_dim = kwargs.get('output_dim', 1)
    input_dim = kwargs.get('inpput_dim', 1)
    self.device = kwargs.get('device','cpu')
    self.dtype = kwargs.get('dtype', torch.float64)
    self.activation = step

    self.weights = nn.Parameter(torch.randn(output_dim, input_dim, dtype = self.dtype, device = self.device) , 
                                  requires_grad = False)

    lsh_batches = lambda input : f_batch_level_map(input, output_dim, self.weights)

    self.batch_level_map = torch.func.vmap(lsh_batches, in_dims=0)


  def forward(self, x, **kwargs):
    return self.activation(self.batch_level_map(x) )

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]

    self.weights = self.weights.to(*args, **kwargs)
    
    return self