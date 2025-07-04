import torch
from torch import nn

from st_nca.common import normalizations

#My own implementation, already tested
#Original source:  https://github.com/petroniocandido/clshq_tk/blob/main/clshq_tk/modules/attention.py


def get_config(model):
  return { 'num_heads': model.num_heads,
          #'num_tokens': model.num_tokens,
          #'embed_dim': model.embed_dim,
          #'device': model.device,
          #'dtype': model.dtype,
          #'normalization': model.normalization.__name__,
          'normalization': model.normalization,
          'pre_norm': model.pre_norm,
          'transformer_feed_forward': model.linear1.weight.size(0),
          #'transformer_activation': model.activation.__class__.__name__
          'transformer_activation': model.activation
            }


def f_token_level(x,y):
  return x @ y

token_level = torch.func.vmap(f_token_level, in_dims=0)

#@torch.compile
def f_sequence_level(x,t,w):
  return token_level(x, w.repeat(t,1,1))

#@torch.compile
def f_batch_level(x, w):
  return token_level(x, w.repeat(x.size(0),1,1))


class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, num_tokens, embed_dim,
               device = None, dtype = torch.float64, **kwargs):
    super().__init__()

    self.num_heads = num_heads
    self.num_tokens = num_tokens
    self.embed_dim = embed_dim
    self.dk = kwargs.get('dk', self.embed_dim)
    self.dv = kwargs.get('dv', self.embed_dim)
    self.device = device
    self.dtype = dtype

    self.sm = nn.Softmax(1)

    self.WQ = [nn.Parameter(torch.randn(self.embed_dim, self.dk, device = self.device, dtype = self.dtype))
               for i in range(self.num_heads)]
    self.WK = [nn.Parameter(torch.randn(self.embed_dim, self.dk, device = self.device, dtype = self.dtype))
               for i in range(self.num_heads)]
    self.WV = [nn.Parameter(torch.randn(self.embed_dim, self.dv, device = self.device, dtype = self.dtype))
               for i in range(self.num_heads)]

    self.WO = nn.Parameter(torch.randn(self.num_heads * self.dv, self.embed_dim, device = self.device, dtype = self.dtype))

  def forward(self, x):
    b, t, e = x.size()

    if t != self.num_tokens:
      raise Exception("Number of tokens different from num_tokens")

    if e != self.embed_dim:
      raise Exception("Token dimension different from embed_dim")

    x = x.to(self.dtype)

    Z = torch.zeros(b, self.num_heads, self.num_tokens, self.dv, device = self.device, dtype = self.dtype)
    Z2 = torch.zeros(b, self.num_tokens, self.embed_dim, device = self.device, dtype = self.dtype)
    for h in range(self.num_heads):
      Q_seq_fun = lambda x : f_sequence_level(x, self.num_tokens, self.WQ[h])
      Q_sequence_level = torch.func.vmap(Q_seq_fun, in_dims=0)
      Q = Q_sequence_level(x)

      K_seq_fun = lambda x : f_sequence_level(x, self.num_tokens, self.WK[h])
      K_sequence_level = torch.func.vmap(K_seq_fun, in_dims=0)
      K = K_sequence_level(x)

      V_seq_fun = lambda x : f_sequence_level(x, self.num_tokens, self.WV[h])
      V_sequence_level = torch.func.vmap(V_seq_fun, in_dims=0)
      V = V_sequence_level(x)

      scores = Q @ K.view(b,e,t)

      A = self.sm(scores / K.size(1) ** 0.5)

      Z[:, h, :, :] = A @ V

    Z_batch_fun = lambda input : f_batch_level(input, self.WO)
    Z_batch_level = torch.func.vmap(Z_batch_fun, in_dims=0)
    Zt = Z.reshape(b, self.num_tokens, self.num_heads * self.dv)
    Z2 = Z_batch_level(Zt)

    return Z2

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)

    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]

    for h in range(self.num_heads):
      self.WQ[h] = self.WQ[h].to(*args, **kwargs)
      self.WK[h] = self.WK[h].to(*args, **kwargs)
      self.WV[h] = self.WV[h].to(*args, **kwargs)
    self.WO = self.WO.to(*args, **kwargs)
    return self
  
  def train(self, *args, **kwargs):
    self = super().train(*args, **kwargs)
    return self
    
 
 #My own implementation, already tested
#Original source:  https://github.com/petroniocandido/clshq_tk/blob/main/clshq_tk/modules/transformer.py

class Transformer(nn.Module):
  def __init__(self, num_heads, num_tokens,  embed_dim, feed_forward, activation = nn.GELU(),
               device = None, dtype=torch.float64, **kwargs):
    super().__init__()

    self.num_heads = num_heads
    self.num_tokens = num_tokens
    self.embed_dim = embed_dim
    self.device = device
    self.dtype = dtype
    self.normalization = kwargs.get('normalization', nn.LayerNorm)
    if isinstance(self.normalization, str):
      self.normalization = normalizations[self.normalization]
    self.pre_norm = kwargs.get('pre_norm', False)
    self.attention = MultiHeadAttention(num_heads, num_tokens, embed_dim,
                            dtype=self.dtype, device=self.device)
    self.ln1 = self.normalization(embed_dim, dtype=self.dtype, device=self.device)
    self.ln2 = self.normalization(embed_dim, dtype=self.dtype, device=self.device)
    self.flat = nn.Flatten(1)
    self.linear1 = nn.Linear(num_tokens * embed_dim, feed_forward,
                            dtype=self.dtype, device=self.device)
    self.linear2 = nn.Linear(feed_forward, num_tokens * embed_dim,
                            dtype=self.dtype, device=self.device)
    self.activation = activation
    self.drop = nn.Dropout(.25)
    self.unflat = nn.Unflatten(1, [num_tokens, embed_dim])

  def forward(self, x):
    if self.pre_norm:
      z = self.ln1(x)
    else:
      z = x
    z = self.attention(z)
    if self.pre_norm:
      z1 = x + z
    else:
      z1 = self.ln1(x + z)
    z = self.flat(z1)
    z = self.activation(self.linear1(self.drop(z)))
    z = self.activation(self.linear2(self.drop(z)))
    z = self.unflat(z)
    z = self.ln2(z + z1)
    return z

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]
    self.attention = self.attention.to(*args, **kwargs)
    self.linear1 = self.linear1.to(*args, **kwargs)
    self.linear2 = self.linear2.to(*args, **kwargs)
    self.ln1 = self.ln1.to(*args, **kwargs)
    self.ln2 = self.ln2.to(*args, **kwargs)
    return self

  def train(self, *args, **kwargs):
    super().train(*args, **kwargs)
    self.attention = self.attention.train(*args, **kwargs)
    self.linear1 = self.linear1.train(*args, **kwargs)
    self.linear2 = self.linear2.train(*args, **kwargs)
    return self