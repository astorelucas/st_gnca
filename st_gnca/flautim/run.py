import flautim as fl
import Dataset, Experiment, Model 
import numpy as np
import pandas as pd
import flautim.metrics as flm


if __name__ == '__main__':

    context = fl.init()

    fl.log(f"Flautim inicializado!!!")
    
    dataset = Dataset.PEMSDataset(data_file = ?, edges_file = ?) # What are the parameters

    model = Model.GNCA(context, input_dim = ?, output_dim = ?, hidden_dim = ?, edge_index =?, 
                       cfg = ?, dropout = ?) # Additional model parameters

    experiment = Experiment.GNCAExperiment(model, dataset, context)

    experiment.run(metrics = {'ACCURACY': flm.Metrics.accuracy, 'ACCURACY_2': flm.Metrics.accuracy_2})