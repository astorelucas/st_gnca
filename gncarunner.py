import torch
from torch import nn
from torch.nn import SmoothL1Loss
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import time
import json

from st_gnca.training.gncatraining import (
    train_gnca_model, 
    plot_training_loss, 
    test_gnca_model
)
from st_gnca.dataloader.database import DataBase, BatchBuilder
from st_gnca.cellmodel.cell_model import xLSTMForecast, LSTMForecast
from xlstm import (xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig,
                     sLSTMBlockConfig, sLSTMLayerConfig, FeedForwardConfig)
from st_gnca.globalmodel.gnca import GraphCellularAutomata
from st_gnca.training.evaluate import HybridLoss


# Setup device and data types
DEVICE = (
    torch.device('cuda') if torch.cuda.is_available()
    else torch.device('mps') if torch.backends.mps.is_available()
    else torch.device('cpu')
)
DTYPE = torch.float32
DEFAULT_PATH = 'st_gnca/'
DATA_PATH = DEFAULT_PATH + 'data/PEMS03/'

# Fixed parameters (DO NOT TUNE)
HORIZON = 12  # Predicting 12 time steps ahead
SEQUENCE_LEN = 24  # Using past 36 time steps


def parse_arguments():
    """Parse command line arguments for save_path, save_suffix, and number of trials."""
    parser = argparse.ArgumentParser(
        description='Train and test GNCA model with Optuna hyperparameter optimization.'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default=DEFAULT_PATH,
        help='Path where models and results will be saved. Default: st_gnca/'
    )
    parser.add_argument(
        '--save_suffix',
        type=str,
        default='__DEFAULT__',
        help='Suffix for saved files. Use "__DEFAULT__" for timestamp. Default: __DEFAULT__'
    )
    parser.add_argument(
        '--n_trials',
        type=int,
        default=10,
        help='Number of Optuna trials to run. Default: 10'
    )
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=1,
        help='Number of parallel jobs for Optuna. Default: 1 (sequential)'
    )
    return parser.parse_args()


def get_save_suffix(save_suffix_arg):
    """Generate or return the save suffix."""
    if save_suffix_arg == '__DEFAULT__':
        return datetime.now().strftime('%Y%m%d_%H%M%S')
    return save_suffix_arg


def ensure_directories(save_path):
    """Create necessary directories if they don't exist."""
    models_dir = Path(save_path) / 'saved_models'
    results_dir = Path(save_path) / 'results'
    optuna_dir = Path(save_path) / 'optuna_studies'
    
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    optuna_dir.mkdir(parents=True, exist_ok=True)
    
    return str(models_dir), str(results_dir), str(optuna_dir)


def create_objective(data, batches, temporal_emb_dim, models_dir, results_dir, save_suffix):
    """
    Create the Optuna objective function.
    
    Args:
        data: DataBase object
        batches: BatchBuilder object
        temporal_emb_dim: Temporal embedding dimension
        models_dir: Directory to save models
        results_dir: Directory to save results
        save_suffix: Suffix for saved files
    
    Returns:
        objective: Function that takes a trial and returns validation loss
    """
    
    def objective(trial):
        """Optuna objective function to minimize validation loss."""
        
        # Start timing
        start_time = time.time()
        
        # Suggest hyperparameters
        hidden_dim = trial.suggest_categorical('hidden_dim', [64, 96, 128, 192])
        gat_heads = trial.suggest_categorical('gat_heads', [2, 4, 8])
        num_layers = trial.suggest_int('num_layers', 4, 12, step=2)
        dropout = trial.suggest_float('dropout', 0.1, 0.3, step=0.05)
        lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        smoothl1_beta = trial.suggest_float('smoothl1_beta', 0.5, 1.0, step=0.1)
        
        # Fixed parameters
        output_dim = HORIZON
        
        # Rebuild batches with new batch_size
        trial_batches = BatchBuilder(
            data, 
            batch_size=batch_size, 
            sequence_len=SEQUENCE_LEN, 
            horizon=HORIZON,
            val_ratio=0.2,
            train_ratio=0.6,
            device=DEVICE,
            dtype=DTYPE
        )
        
        # Calculate feature dimension
        feature_dim = temporal_emb_dim + ((2 * hidden_dim * gat_heads))
        
        # Initialize cell model
        cell_model = LSTMForecast(
            feature_dim=feature_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            edge_index=data.edge_index,
            graph=data.G,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Initialize GNCA model
        gnca = GraphCellularAutomata(
            graph=data.G,
            cell_model=cell_model,
            device=DEVICE,
            dtype=DTYPE,
            temp_dim=temporal_emb_dim,
            heads=gat_heads,
            laplacian_components=36,
            dropout=dropout
        )
        
        # Move model to device
        gnca = gnca.to(DEVICE)
        
        # Initialize optimizer and criterion
        optimizer = torch.optim.AdamW(
            gnca.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        criterion = SmoothL1Loss(beta=smoothl1_beta)
        
        # Construct save paths for this trial
        model_save_path = str(
            Path(models_dir) / f'gnca_trial_{trial.number}_{save_suffix}.pth'
        )
        
        # Train model with early stopping (reduced epochs for efficiency)
        try:
            avg_loss, training_losses = train_gnca_model(
                gnca, 
                trial_batches.get_train_loader(), 
                optimizer=optimizer, 
                criterion=criterion,
                num_epochs=30,  # You can adjust this if needed
                device=DEVICE,
                return_history=True,
                save_path=model_save_path,
                scaler=data.value_embedding.embedder,
                val_loader=trial_batches.get_val_loader()
            )
            
            # Get final validation loss
            val_loss = avg_loss  # This should be the best validation loss from training
            
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            val_loss = float('inf')
        
        # Calculate trial duration
        trial_duration = time.time() - start_time
        
        # Store trial duration and other metadata
        trial.set_user_attr('duration_seconds', trial_duration)
        trial.set_user_attr('feature_dim', feature_dim)
        
        # Clean up to save memory
        del gnca, cell_model, optimizer, trial_batches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return val_loss
    
    return objective


def save_study_results(study, save_path, save_suffix):
    """Save Optuna study results to files."""
    
    # Save best parameters
    best_params_path = Path(save_path) / f'best_params_{save_suffix}.json'
    with open(best_params_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    
    # Save all trials information
    trials_df = study.trials_dataframe()
    trials_csv_path = Path(save_path) / f'all_trials_{save_suffix}.csv'
    trials_df.to_csv(trials_csv_path, index=False)
    
    # Create summary
    summary = {
        'best_trial_number': study.best_trial.number,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'total_trials': len(study.trials),
        'total_duration_seconds': sum(
            t.user_attrs.get('duration_seconds', 0) for t in study.trials
        ),
        'average_trial_duration_seconds': sum(
            t.user_attrs.get('duration_seconds', 0) for t in study.trials
        ) / len(study.trials) if study.trials else 0
    }
    
    summary_path = Path(save_path) / f'study_summary_{save_suffix}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print("\n" + "="*80)
    print("OPTUNA STUDY SUMMARY")
    print("="*80)
    print(f"Best trial: {summary['best_trial_number']}")
    print(f"Best validation loss: {summary['best_value']:.6f}")
    print(f"Total trials: {summary['total_trials']}")
    print(f"Total duration: {summary['total_duration_seconds']:.2f} seconds")
    print(f"Average trial duration: {summary['average_trial_duration_seconds']:.2f} seconds")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*80)
    
    return summary


if __name__ == "__main__":
    print("Parsing command line arguments...")
    args = parse_arguments()
    
    # Get save suffix (with timestamp if __DEFAULT__)
    save_suffix = get_save_suffix(args.save_suffix)
    save_path = args.save_path
    n_trials = args.n_trials
    n_jobs = args.n_jobs
    
    print(f"Save path: {save_path}")
    print(f"Save suffix: {save_suffix}")
    print(f"Number of trials: {n_trials}")
    print(f"Parallel jobs: {n_jobs}")
    
    # Create directories
    models_dir, results_dir, optuna_dir = ensure_directories(save_path)
    
    print("\nLoading data...")
    
    # Load data
    try:
        data = DataBase(
            edges_file=DATA_PATH + 'edges_normalized.csv',
            data_file=DATA_PATH + 'data_imputed.csv'
        )
    except Exception as e:
        data = DataBase(
            edges_file=DATA_PATH + 'edges.csv',
            data_file=DATA_PATH + 'data.csv'
        )
    
    print("DataBase initialized.")
    print(f"Fixed horizon: {HORIZON}")
    print(f"Fixed sequence length: {SEQUENCE_LEN}")
    
    # Create initial batches (will be recreated in each trial with different batch_size)
    batches = BatchBuilder(
        data, 
        batch_size=32,  # Default, will be overridden
        sequence_len=SEQUENCE_LEN, 
        horizon=HORIZON,
        val_ratio=0.2,
        train_ratio=0.6,
        device=DEVICE,
        dtype=DTYPE
    )
    
    print("BatchBuilder initialized.")
    
    # Get temporal embedding dimension
    temporal_emb_dim = data.temporal_features.size(1)
    
    print("\n" + "="*80)
    print("STARTING OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    
    # Create Optuna study
    sampler = TPESampler(seed=42)  # For reproducibility
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=10)
    
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
        study_name=f'gnca_optimization_{save_suffix}'
    )
    
    # Create objective function
    objective_fn = create_objective(
        data, 
        batches, 
        temporal_emb_dim, 
        models_dir, 
        results_dir, 
        save_suffix
    )
    
    # Run optimization
    study.optimize(
        objective_fn, 
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True
    )
    
    # Save results
    summary = save_study_results(study, optuna_dir, save_suffix)
    
    # Optionally train final model with best parameters
    print("\n" + "="*80)
    print("Training final model with best parameters...")
    print("="*80)
    
    best_params = study.best_params
    
    # Rebuild batches with best batch_size
    final_batches = BatchBuilder(
        data, 
        batch_size=best_params['batch_size'], 
        sequence_len=SEQUENCE_LEN, 
        horizon=HORIZON,
        val_ratio=0.2,
        train_ratio=0.6,
        device=DEVICE,
        dtype=DTYPE
    )
    
    # Calculate feature dimension
    feature_dim = temporal_emb_dim + (
        (2 * best_params['hidden_dim'] * best_params['gat_heads'])
    )
    
    # Initialize best cell model
    best_cell_model = LSTMForecast(
        feature_dim=feature_dim,
        output_dim=HORIZON,
        hidden_dim=best_params['hidden_dim'],
        edge_index=data.edge_index,
        graph=data.G,
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout']
    )
    
    # Initialize best GNCA model
    best_gnca = GraphCellularAutomata(
        graph=data.G,
        cell_model=best_cell_model,
        device=DEVICE,
        dtype=DTYPE,
        temp_dim=temporal_emb_dim,
        heads=best_params['gat_heads'],
        laplacian_components=36,
        dropout=best_params['dropout']
    )
    
    # Train final model
    final_model_path = str(Path(models_dir) / f'gnca_best_model_{save_suffix}.pth')
    final_loss_plot_path = str(Path(results_dir) / f'gnca_best_training_loss_{save_suffix}.png')
    final_test_results_path = str(Path(results_dir) / f'gnca_best_test_results_{save_suffix}.pth')
    
    avg_loss, training_losses = train_gnca_model(
        best_gnca, 
        final_batches.get_train_loader(), 
        optimizer=torch.optim.AdamW(
            best_gnca.parameters(), 
            lr=best_params['lr'], 
            weight_decay=best_params['weight_decay']
        ), 
        criterion=SmoothL1Loss(beta=best_params['smoothl1_beta']),
        num_epochs=30,
        device=DEVICE,
        return_history=True,
        save_path=final_model_path,
        scaler=data.value_embedding.embedder,
        val_loader=final_batches.get_val_loader()
    )
    
    print("Final model training completed.")
    
    # Plot final training loss
    plot_training_loss(
        training_losses,
        save_path=final_loss_plot_path,
        show=False
    )
    
    # Test final model
    results = test_gnca_model(
        best_gnca, 
        final_batches.get_test_loader(), 
        temp_dim=temporal_emb_dim,
        device=DEVICE,
        save_predictions_path=final_test_results_path,
        scaler=data.value_embedding.embedder
    )
    
    print("\n" + "="*80)
    print("FINAL MODEL TEST RESULTS")
    print("="*80)
    print(results)
    print("="*80)
    
    print("\nOptimization completed successfully!")
    print(f"All results saved to: {save_path}")