from flautim.pytorch.centralized.Experiment import Experiment
import flautim as fl
import numpy as np
import torch
import time

class GNCAExperiment(Experiment):
    def __init__(self, model, dataset, context, **kwargs):
        super(GNCAExperiment, self).__init__(model, dataset, context, **kwargs)


    def training_loop(self, data_loader):
        pass

    def validation_loop(self, data_loader):
        pass