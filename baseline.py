#!/usr/bin/env python3
"""
baseline.py - Model Comparison Baseline Script

This script tests all available models with MSE loss and a forecast horizon of 10 steps
on a single CSV file. It provides a baseline comparison of model performance.
"""

import os
import sys
import glob
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import time
from tqdm import tqdm
import pandas as pd

# Append parent directory to locate modules (assumes a project folder structure)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your custom dataset and trainer.
from utils.datasets import EMG_dataset
from utils.Trainer import Trainer  

# Import the summary and logging utilities from metrics.py
from utils.metrics import save_summary_csv  # and log_epoch_loss if needed inside Trainer

# Import model variants.
from models.models import (
    LSTMModel, RNNModel, GRUModel, TCNModel, TemporalTransformer,
    TimeSeriesTransformer, Informer, NBeats, DBN,
    PatchTST, CrossFormer, DLinear
)

# ----------------------------------------------------------------------------------
# Utility function to generate trial directories.
# ----------------------------------------------------------------------------------
def get_trial_dir(model_name):
    trial_dir = os.path.join(BASE_TRIAL_DIR, model_name)
    os.makedirs(trial_dir, exist_ok=True)
    return trial_dir

# ----------------------------------------------------------------------------------
# Main training function for a single CSV file
# ----------------------------------------------------------------------------------
def run_training(model_class, model_name, csv_path, epochs, sensor_mode="emg", target_sensor="emg"):
    print(f"Starting training: {model_name} using input: {sensor_mode} with MSE loss")
    
    # Generate trial directories
    trial_dir = get_trial_dir(model_name)
    MODELS_DIR = os.path.join(trial_dir, "models")
    LOGS_DIR = os.path.join(trial_dir, "logs")
    FIGURES_DIR = os.path.join(trial_dir, "figures")
    for d in [MODELS_DIR, LOGS_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)

    # Determine if we should keep time data (only for Informer model)
    keep_time = model_name.lower() == "informer"

    # Get the number of channels based on input sensor type.
    selected_channels = input_sizes[sensor_mode]

    # Create dataset from single file
    dataset = EMG_dataset(
        processed_index_csv=csv_path,
        lag=lag,
        n_ahead=n_ahead,
        input_sensor=sensor_mode,
        target_sensor=target_sensor,
        base_dir=base_dir,
        single_file_mode=True,
        keep_time=keep_time
    )
    
    # Split dataset into train and validation (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Instantiate the model based on the model name.
    if model_name == "tcn":
        model = model_class(
            input_channels=selected_channels, 
            num_classes=output_size,
            n_ahead=n_ahead
        ).to(device)
    elif model_name == "timeseries_transformer":
        model = model_class(
            input_size=selected_channels,
            output_size=output_size,
            n_ahead=n_ahead
        ).to(device)
    elif model_name == "temporal_transformer":
        model = model_class(
            input_size=selected_channels,
            num_classes=output_size,
            n_ahead=n_ahead
        ).to(device)
    elif model_name == "nbeats":
        flattened_input_size = lag * selected_channels
        model = model_class(
            input_size=flattened_input_size,
            output_size=output_size,
            stack_types=["trend", "seasonality"],
            num_blocks_per_stack=3,
            n_ahead=n_ahead
        ).to(device)
    elif model_name == "dbn":
        # Create a list of layer sizes for the DBN
        layer_sizes = [selected_channels * lag, 128, 64, 32]  
        model = model_class(
            sizes=layer_sizes,
            output_dim=output_size,
            k=1,
            rbm_lr=1e-4,
            n_ahead=n_ahead
        ).to(device)
    elif model_name == "informer":
        model = model_class(
            enc_in=selected_channels,  # Input dimension
            dec_in=selected_channels,  # Decoder input dimension
            c_out=output_size,         # Output dimension
            seq_len=lag,               # Input sequence length
            label_len=lag // 2,        # Label length (half of input length)
            out_len=n_ahead,           # Prediction length
            factor=5,                  # ProbSparse attention factor
            d_model=512,               # Dimension of model
            n_heads=8,                 # Number of heads
            e_layers=3,                # Number of encoder layers
            d_layers=2,                # Number of decoder layers
            d_ff=512,                  # Dimension of FCN
            dropout=0.0,               # Dropout rate
            attn='prob',               # Attention type
            embed='fixed',             # Embedding type
            freq='h',                  # Frequency for time features
            activation='gelu',         # Activation function
            output_attention=False,    # Whether to output attention
            distil=True,               # Whether to use distilling in encoder
            mix=True,                  # Whether to use mix attention
            device=device              # Device
        ).to(device)
    elif model_name == "patchtst":
        model = model_class(
            input_channels=selected_channels,
            patch_size=16,
            d_model=512,
            nhead=8,
            num_layers=3,
            dim_feedforward=2048,
            dropout=0.1,
            forecast_horizon=n_ahead,
            output_size=output_size
        ).to(device)
    elif model_name == "crossformer":
        model = model_class(
            input_channels=selected_channels,
            seq_len=lag,
            d_model=64,
            nhead=8,
            num_layers=3,
            dim_feedforward=256,
            dropout=0.1,
            forecast_horizon=n_ahead,
            output_size=output_size
        ).to(device)
    elif model_name == "dlinear":
        model = model_class(
            seq_len=lag,
            forecast_horizon=n_ahead,
            num_channels=selected_channels,
            individual=False,
            moving_avg_kernel=25
        ).to(device)
    else:
        # Default model initialization for LSTM, GRU, RNN
        model = model_class(
            input_size=selected_channels,
            hidden_size=512,
            num_classes=output_size,
            num_layers=3,
            n_ahead=n_ahead
        ).to(device)

    # Use MSE loss for all models
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

    epoch_log_file = os.path.join(LOGS_DIR, "epoch_log.csv")
    # Create trainer instance; note that Trainer will use the log_epoch_loss function from metrics.py
    trainer = Trainer(
        model=model,
        lag=lag,
        n_ahead=n_ahead,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        testloader=val_loader,  # Using validation set as test set for baseline
        fig_dir=FIGURES_DIR,
        loss='mse',
        model_type=model_name,
        device=device,
        clip_grad_norm=1.0,
        sensor_mode=sensor_mode,
        epoch_log_file=epoch_log_file
    )

    # Check if checkpoint exists and load it
    checkpoint_path = os.path.join(trial_dir, "models", "checkpoint.pth")
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch}")
        epochs -= start_epoch

    # Train model
    start_time = time.time()
    metrics = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        checkpoint_dir=MODELS_DIR,
        patience=10,
        min_delta=0.0,
        num_windows=5,
        loss_curve_path=os.path.join(MODELS_DIR, "loss_curve.png")
    )
    training_time = time.time() - start_time
    
    # Save the final model
    final_model_path = os.path.join(MODELS_DIR, f"{model_name}_final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    # Evaluate the model on the validation set
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in val_loader:
            if model_name.lower() == "informer" and len(batch) > 4:
                X, Y, X_time, Y_time, actions, weights = batch
                X = X.to(device)
                Y = Y.to(device)
                X_time = X_time.to(device) if X_time is not None else None
                Y_time = Y_time.to(device) if Y_time is not None else None
                label_len = lag // 2
                dec_inp = torch.zeros((X.size(0), n_ahead, X.size(2)), device=device)
                if label_len > 0:
                    dec_inp = torch.cat([Y[:, :label_len, :], dec_inp], dim=1)
                if X_time is not None:
                    encoder_seq_len = X.size(1)
                    if X_time.size(1) != encoder_seq_len:
                        if X_time.size(1) < encoder_seq_len:
                            padding_len = encoder_seq_len - X_time.size(1)
                            last_time_entries = X_time[:, -1:, :].repeat(1, padding_len, 1)
                            enc_time_mark = torch.cat([X_time, last_time_entries], dim=1)
                        else:
                            enc_time_mark = X_time[:, :encoder_seq_len, :]
                    else:
                        enc_time_mark = X_time
                else:
                    enc_time_mark = torch.zeros((X.size(0), X.size(1), 5), device=device).long()
                
                decoder_seq_len = dec_inp.size(1)
                if Y_time is not None:
                    if Y_time.size(1) < decoder_seq_len:
                        padding_len = decoder_seq_len - Y_time.size(1)
                        last_time_entries = Y_time[:, -1:, :].repeat(1, padding_len, 1)
                        dec_time_mark = torch.cat([Y_time, last_time_entries], dim=1)
                    else:
                        dec_time_mark = Y_time[:, :decoder_seq_len, :]
                else:
                    dec_time_mark = torch.zeros((X.size(0), decoder_seq_len, 5), device=device).long()
                predictions = model(X, enc_time_mark, dec_inp, dec_time_mark)
            else:
                X, Y = batch[0], batch[1]
                X = X.to(device)
                Y = Y.to(device)
                predictions = model(X)
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(Y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    final_mse = np.mean((all_preds - all_targets) ** 2)
    
    # Plot sample predictions.
    sample_idx = np.random.randint(0, len(all_preds))
    plt.figure(figsize=(12, 8))
    for i in range(output_size):
        plt.subplot(output_size, 1, i+1)
        plt.plot(all_targets[sample_idx, :, i], 'b-', label=f'Actual Channel {i+1}')
        plt.plot(all_preds[sample_idx, :, i], 'r--', label=f'Predicted Channel {i+1}')
        plt.legend()
        plt.grid(True)
    plt.suptitle(f'{model_name.upper()} Sample Predictions')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"{model_name}_sample_predictions.png"))
    plt.close()
    
    # Compute additional evaluation metrics using the trainer's function.
    additional_metrics = trainer.compute_metrics(val_loader)
    
    # Return performance metrics including extra metrics for summary CSV.
    return {
        'model': model_name,
        'final_val_mse': final_mse,
        'best_val_mse': min(metrics['val_loss']),
        'training_time': training_time,
        'epochs_trained': len(metrics['train_loss']),
        'rmse': additional_metrics['rmse'],
        'mae': additional_metrics['mae'],
        'r2_score': additional_metrics['r2_score'],
        'corr_coef': additional_metrics['corr_coef']
    }

def test_model(model_name, model_class, csv_path, epochs):
    """Wrapper function to run training with proper logging."""
    print(f"\nStarting training for: {model_name}")
    return run_training(model_class, model_name, csv_path, epochs)

# ----------------------------------------------------------------------------------
# DEVICE / PATH CONFIGURATION & HYPERPARAMETERS
# ----------------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the base folder for trials.
BASE_TRIAL_DIR = "/data1/dnicho26/Thesis/AI-Assisted-Gait-Restoration-for-Disabled-Individuals/baseline_trials"
os.makedirs(BASE_TRIAL_DIR, exist_ok=True)

# Define the base directory for your dataset.
base_dir = "/data1/dnicho26/EMG_DATASET/final-data/"

# Fixed parameters for baseline testing
lag = 30         # Length of the sliding window.
n_ahead = 10     # Forecast horizon of 10
batch_size = 12
epochs = 300
learning_rate = 1e-4
output_size = 3
target_sensor = "emg"  

# Define input sizes (number of channels) for each sensor mode.
input_sizes = {"all": 21, "emg": 3, "acc": 9, "gyro": 9}

# ----------------------------------------------------------------------------------
# Define model variants.
# ----------------------------------------------------------------------------------
model_variants = {
    "informer": Informer,
    "patchtst": PatchTST,
    "crossformer": CrossFormer,
    "dlinear": DLinear,
    "timeseries_transformer": TimeSeriesTransformer,
    "temporal_transformer": TemporalTransformer,
    "dbn": DBN,
    "nbeats": NBeats,
    "tcn": TCNModel,
    "lstm": LSTMModel,
    "rnn": RNNModel,
    "gru": GRUModel,
}

# ----------------------------------------------------------------------------------
# Main function
# ----------------------------------------------------------------------------------
def main():
    print(f"Starting baseline testing on device: {device}")
    print(f"Testing with n_ahead = {n_ahead}")
    
    # Find a single CSV file to use for testing.
    csv_path = "/data1/dnicho26/EMG_DATASET/final-data/1/walk and turn left/1738795743.3349926.csv"
    
    # Ensure the file exists.
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    print(f"Using CSV file: {csv_path}")
    
    # Test each model and collect results.
    results = []
    for model_name, model_class in model_variants.items():
        model_results = test_model(model_name, model_class, csv_path, epochs)
        results.append(model_results)
        print(f"Completed testing {model_name}")
    
    # Save the summary CSV using our metrics utility function.
    save_summary_csv(results, BASE_TRIAL_DIR)
    
    # Optionally, create a comparison bar chart.
    results_df = pd.DataFrame(results).sort_values('final_val_mse')
    plt.figure(figsize=(12, 8))
    plt.bar(results_df['model'], results_df['final_val_mse'])
    plt.title('Model Comparison - Validation MSE')
    plt.xlabel('Model')
    plt.ylabel('MSE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_TRIAL_DIR, "model_comparison.png"))
    
    print("\nBaseline testing completed!")
    print(f"Results saved to {os.path.join(BASE_TRIAL_DIR, 'baseline_results.csv')}")
    print("\nModel Ranking (by validation MSE):")
    for i, (_, row) in enumerate(results_df.iterrows()):
        print(f"{i+1}. {row['model']} - MSE: {row['final_val_mse']:.6f} - Time: {row['training_time']:.2f}s")

if __name__ == "__main__":
    main()
