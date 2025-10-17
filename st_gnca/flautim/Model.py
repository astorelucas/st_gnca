from flautim.pytorch.Model import Model
import torch
from st_gnca.cellmodel.cell_model import xLSTMForecast

class GNCA(Model):
    def __init__(self, context, **kwargs):
        super(GNCA, self).__init__(context, name = "GNCA-CellModel", **kwargs)

        input_dim = kwargs.get("input_dim", 1)
        output_dim = kwargs.get("output_dim", 1)
        hidden_dim = kwargs.get("hidden_dim", 1)
        edge_index = kwargs.get("edge_index", 1)
        cfg = kwargs.get("cfg", None)
        dropout = kwargs.get("dropout", 0.15)

        self.model = xLSTMForecast(input_dim, output_dim, hidden_dim, edge_index, cfg, dropout)

    def forward(self, x):
        return self.model.forward(x)
    
    def parameters(self):
        return self.model.parameters()
    
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.model = self.model.to(*args, **kwargs)
        return self

    def train(self, *args, **kwargs):
        self = super().train(*args, **kwargs)
        self.model = self.model.train(*args, **kwargs)
        return self
    
