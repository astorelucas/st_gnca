from flautim.pytorch.centralized.Experiment import Experiment
import torch
import torch.nn as nn

class GNCAExperiment(Experiment):
    def __init__(self, model, dataset, context, **kwargs):
        super(GNCAExperiment, self).__init__(model, dataset, context, **kwargs)
        
        # Inicializar função de perda e otimizador
        self.loss_fn = nn.MSELoss()  # Para regressão de séries temporais
        learning_rate = kwargs.get('learning_rate', 0.001)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def training_loop(self, data_loader):
        """Loop de treinamento iterando sobre batches"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            # Mover dados para o dispositivo (GPU/CPU)
            inputs = inputs.to(self.context.device)
            targets = targets.to(self.context.device)
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Calcular perda
            loss = self.loss_fn(outputs, targets)
            
            # Backward pass e otimização
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss

    def validation_loop(self, data_loader):
        """Loop de validação para avaliar o modelo"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs = inputs.to(self.context.device)
                targets = targets.to(self.context.device)
                
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
