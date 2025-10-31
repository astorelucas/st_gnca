import flautim as fl
import Dataset, Experiment, Model 
import pandas as pd
import torch

if __name__ == '__main__':
    context = fl.init()
    fl.log("Flautim inicializado!")
    
    # ========== PARÂMETROS DO DATASET ==========
    data_file = "data.csv"
    edges_file = "edges.csv"
    
    fl.log("Carregando dataset...")
    dataset = Dataset.PEMSDataset(
        data_file=data_file,
        edges_file=edges_file,
        batch_size=32,
        sequence_len=12,
        train_split=0.7,
        val_split=0.1
    )
    fl.log("Dataset carregado!")
    # ========== PREPARAR EDGE_INDEX ==========
    # Carregar arquivo de edges e converter para formato COO [2, num_edges]
    fl.log("Preparando edge_index...")
    edges_df = pd.read_csv(edges_file)

    fl.log('edges_df carregado:'+ str(edges_df.shape))

    edge_index = torch.tensor([
        edges_df['source'].values,
        edges_df['target'].values
    ], dtype=torch.long)
    fl.log("edge_index preparado!")


    fl.log("Iniciando modelo...")
    # ========== PARÂMETROS DO MODELO ==========
    model = Model.GNCA(
        context,
        input_dim=3,        # velocidade, fluxo, ocupação
        output_dim=1,       # velocidade prevista
        hidden_dim=64,
        edge_index=edge_index,
        cfg=None,
        dropout=0.15
    )
    fl.log("Modelo iniciado!")
    
    # ========== EXECUTAR EXPERIMENTO ==========
    fl.log("Iniciando experimento...")
    experiment = Experiment.GNCAExperiment(
        model, dataset, context, learning_rate=0.001
    )
    fl.log("Experimento iniciado!")

    fl.log("Executando experimento...")
    experiment.run(metrics={'MSE': fl.metrics.mse, 'MAE': fl.metrics.mae})
    fl.log("Experimento concluído!")