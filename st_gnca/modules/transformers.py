import torch
from torch import nn

from st_gnca.common import normalizations

class Transformer(nn.Module):
    """
    Adapter around nn.TransformerEncoder that mimics the public interface
    expected by CellModel.
    """
    def __init__(
        self,
        num_heads: int,
        num_tokens: int,          # ignored by nn.Transformer, kept for signature parity
        dim_token: int,
        feedforward_dim: int,
        activation,
        *,                         # keep kwargs open so CellModel **kwargs works
        dtype=torch.float32,
        device=None,
        **_
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_token,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            activation='gelu',
            batch_first=True,      # (batch, seq, dim) -> matches CellModel
            dtype=dtype,
            device=device,
        )
        
        self.net = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x):
        # torch’s encoder expects (batch, seq, dim) when batch_first=True, so no permute
        return self.net(x)

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

