import numpy as np
import torch
from torch import nn

from st_gnca.modules.lsh import LSH

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Expert(nn.Module):
    def __init__(self, **kwargs):
        super(Expert, self).__init__()
        self.device = kwargs.get('device',DEVICE)
        self.dtype = kwargs.get('dtype',2)
        num_layers = kwargs.get('num_layers',2)
        input_dim = kwargs.get('input_dim',2)
        hidden_dim = kwargs.get('hidden_dim',2)
        output_dim = kwargs.get('output_dim',2)

        dropout_pct = kwargs.get('dropout',0.15)

        self.dropout = nn.Dropout(dropout_pct)

        self.normalization = kwargs.get('layernorm',None)

        self.activation = kwargs.get('activation',nn.GELU())

        if num_layers == 1:
            self.layers = nn.ModuleList( [nn.Linear(input_dim, output_dim, device=self.device, dtype=self.dtype)])
        else:
            self.layers = nn.ModuleList( [nn.Linear(input_dim, hidden_dim, device=self.device, dtype=self.dtype)])
            for k in range(1,num_layers-1):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim, device=self.device, dtype=self.dtype))
            self.layers.append(nn.Linear(hidden_dim, output_dim, device=self.device, dtype=self.dtype))


    def forward(self, x):
        for layer in self.layers:
            if self.normalization is not None:
                x = self.normalization(x)
            x = self.activation(layer(self.dropout(x)))
        return x

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        if isinstance(args[0], str):
            self.device = args[0]
        else:
            self.dtype = args[0]
        self.layers = self.layers.to(*args, **kwargs)
        return self

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.layers = self.layers.train(*args, **kwargs)
        return self


class Router(nn.Module):
    def __init__(self, **kwargs):
        super(Router, self).__init__()
        self.device = kwargs.get('device',DEVICE)
        self.dtype = kwargs.get('dtype',2)

        self.activate = kwargs.get('activate',1)

        input_dim = kwargs.get('input_dim',2)
        hidden_dim = kwargs.get('hidden_dim',2)
        num_experts = kwargs.get('num_experts',2)

        self.dropout = nn.Dropout(0.15)

        self.activation = kwargs.get('activation',nn.GELU())

        # Layers
        self.layer1 = nn.Linear(input_dim, hidden_dim, device=self.device, dtype=self.dtype)
        self.layer2 = nn.Linear(hidden_dim, num_experts, device=self.device, dtype=self.dtype)


    def forward(self, x):
        x = self.activation(self.layer1(self.dropout(x)))
        x = self.activation(self.layer2(self.dropout(x)))

        vals, indexes = torch.topk(x, k=self.activate, dim=1)

        weights = torch.softmax(vals, dim=1)

        return weights, indexes
    
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        if isinstance(args[0], str):
            self.device = args[0]
        else:
            self.dtype = args[0]
        self.layer1 = self.layer1.to(*args, **kwargs)
        self.layer2 = self.layer2.to(*args, **kwargs)
        return self

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.layer1 = self.layer1.train(*args, **kwargs)
        self.layer2 = self.layer2.train(*args, **kwargs)
        return self
    

def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)


class LSHRouter(nn.Module):
    def __init__(self, **kwargs):
        super(LSHRouter, self).__init__()
        self.input_dim = kwargs.get('input_dim',1)
        self.output_dim = kwargs.get('output_dim',1)
        self.lsh = LSH(**kwargs)

    def forward(self, x):
        batch = x.size(0)
        x = self.lsh(x)
        return torch.ones(batch, device=x.device, dtype=x.dtype), bin2dec(x, self.output_dim)
    

class SparseMixtureOfExperts(nn.Module):
    def __init__(self, **kwargs):
        super(SparseMixtureOfExperts, self).__init__()
        self.device = kwargs.get('device',DEVICE)
        self.dtype = kwargs.get('dtype',2)
        self.num_experts = kwargs.get('num_experts',4)
        self.activate = kwargs.get('activate',1)

        self.input_dim = kwargs.get('input_dim',1)
        self.output_dim = kwargs.get('output_dim',1)

        router_hidden_dim = kwargs.get('router_hidden_dim',1)

        expert_hidden_dim = kwargs.get('expert_hidden_dim',1)

        self.experts = nn.ModuleList([
            Expert(hidden_dim = expert_hidden_dim, **kwargs)
            for k in range(self.num_experts)
        ])

        self.router_type = kwargs.get('router','lsh')

        if self.router_type == 'mlp':
            self.router = Router(hidden_dim = router_hidden_dim,**kwargs)
        elif self.router_type == 'lsh': 
            rin_dim = kwargs.pop('router_input_dim',1)
            self.router = LSHRouter(
                input_dim = rin_dim,
                output_dim = int(np.log2(self.num_experts))
                )

    def forward(self, x):
        if self.router_type == 'lsh': 
            weights, routes = self.router(x[:,:self.router.input_dim])
        else:
            weights, routes = self.router(x)

        print(torch.unique(routes, return_counts=True))
        #print(x.shape)

        batch = x.size(0)

        output = torch.zeros(batch, self.output_dim, device=self.device, dtype=self.dtype)

        for ix, expert in enumerate(self.experts):
            filter = (routes == ix).any(dim=-1)
            selected_x = x[filter,:]
            out = expert(selected_x)
            output[filter, :] += out

        return output