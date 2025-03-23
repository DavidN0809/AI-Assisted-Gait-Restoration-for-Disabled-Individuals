#!/usr/bin/env python
"""
Combined training, checkpointing, evaluation, and inference script.

This script:
 • Creates a new run folder inside the Thesis directory.
 • Iterates over different sensor input configurations (e.g. "all", "emg", etc.)
 • For each sensor input, loads train/val/test datasets (which now internally handle scaling).
 • Then, for each model variant (BasicLSTM, BidirectionalLSTM, ResidualLSTM, 
   AttentionLSTM, and TimeSeriesTransformer), it:
     - Initializes the model with appropriate hyperparameters.
     - Resumes from a checkpoint if available.
     - Trains the model (saving a checkpoint every epoch).
     - Saves the final trained model (including the model type in the filename).
     - Evaluates the model on the test set.
     - Plots one test sample’s predicted vs. actual outputs, labeling each sensor channel.
"""

import os
import sys
import glob
import re
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Append parent directory to locate modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.datasets import EMG_dataset
from models.models import BasicLSTM, BidirectionalLSTM, ResidualLSTM, AttentionLSTM, TimeSeriesTransformer, CNNLSTM
from utils.Trainer import ModelTrainer, LogCoshLoss

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

###############################################################################
# 1. Set up the new run directory (with nested subfolders for models, logs, and figures)
###############################################################################
scaler_path = SAVE_BASE_DIR = "/data1/dnicho26/Thesis/AI-Assisted-Gait-Restoration-for-Disabled-Individuals/models/scaler"
SAVE_BASE_DIR = "/data1/dnicho26/Thesis/AI-Assisted-Gait-Restoration-for-Disabled-Individuals/runs"
os.makedirs(SAVE_BASE_DIR, exist_ok=True)

def plot_long_horizon_segments(actual_full, pred_full, segment_length, save_dir, variant_name, input_mode, output_size):
    """
    Plots long-horizon predictions in segments and saves each figure.
    
    Parameters:
        actual_full (ndarray): The ground truth values (shape: [total_timesteps, output_size]).
        pred_full (ndarray): The predicted values (shape: [total_timesteps, output_size]).
        segment_length (int): The number of timesteps in each segment.
        save_dir (str): Directory where the plots will be saved.
        variant_name (str): Name of the model variant (used for plot title and filename).
        input_mode (str): Sensor input configuration name.
        output_size (int): Number of output channels.
    """
    num_segments = len(pred_full) // segment_length
    for seg in range(num_segments):
        start = seg * segment_length
        end = start + segment_length
        
        plt.figure(figsize=(12, 6))
        for ch in range(output_size):
            plt.plot(actual_full[start:end, ch], label=f"Sensor {ch} Actual")
            plt.plot(pred_full[start:end, ch], label=f"Sensor {ch} Predicted", linestyle="--")
        plt.xlabel("Time Step")
        plt.ylabel("Signal Value")
        plt.title(f"Prediction vs Actual (Segment {seg+1}) for {variant_name} ({input_mode})")
        plt.legend()
        
        seg_fig_path = os.path.join(save_dir, f"long_prediction_segment_{seg+1}.png")
        plt.savefig(seg_fig_path)
        plt.close()
        print(f"Saved segment plot to {seg_fig_path}")


# For training, we use HuberLoss here
#loss = nn.MSELoss()
#loss = nn.SmoothL1Loss()
loss = nn.HuberLoss()
#loss = LogCoshLoss()

def get_next_run_dir(base_dir, loss_name):
    existing_runs = [d for d in os.listdir(base_dir) if d.startswith("run") and os.path.isdir(os.path.join(base_dir, d))]
    if existing_runs:
        run_numbers = [int(r.replace("run", "").split("_")[0]) for r in existing_runs if r.replace("run", "").split("_")[0].isdigit()]
        next_run = max(run_numbers) + 1
    else:
        next_run = 1
    run_dir = os.path.join(base_dir, f"run{next_run}_{loss_name}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

run_dir = get_next_run_dir(SAVE_BASE_DIR, "HuberLoss")
print("Run folder:", run_dir)

# Subdirectories for models, logs, and figures
MODELS_DIR    = os.path.join(run_dir, "models")
LOGS_DIR      = os.path.join(run_dir, "logs")
FIGURES_DIR   = os.path.join(run_dir, "figures")
CHECKPOINT_DIR= os.path.join(MODELS_DIR, "checkpoints")
TRAINED_DIR   = os.path.join(MODELS_DIR, "trained")
FIG_TRAIN_DIR = os.path.join(FIGURES_DIR, "training")

for d in [MODELS_DIR, LOGS_DIR, FIGURES_DIR, CHECKPOINT_DIR, TRAINED_DIR, FIG_TRAIN_DIR]:
    os.makedirs(d, exist_ok=True)

###############################################################################
# 2. Define hyperparameters, dataset paths, and input configurations
###############################################################################
# Dataset indices (adjust paths if needed)
base_dir="/data1/dnicho26/EMG_DATASET/data/final-proc-server/index_files"
train_index = f"{base_dir}/index_train_treadmill.csv"
val_index   = f"{base_dir}/index_val_treadmill.csv"
test_index  = f"{base_dir}/index_test_treadmill.csv"
# Hyperparameters
lag         = 60
n_ahead     = 10
batch_size  = 128
epochs      = 150
lr          = 0.0007
hidden_size = 128
num_layers  = 5
output_size = 3   # Always predict 3 channels (target leg EMG)

# Input sensor configurations
input_configs = [ "all", "emg", "acc", "gyro" ]
# These values are used to initialize the models (i.e. the number of input channels)
input_sizes   = {"all": 21, "emg": 3, "acc": 9, "gyro": 9}

# Transformer-specific hyperparameters
num_heads              = 4
transformer_layers     = 2
transformer_hidden_dim = 128
dropout                = 0.1

###############################################################################
# 3. Define model variants to test
###############################################################################

model_variants = {
        "TimeSeriesTransformer": {
        "class": TimeSeriesTransformer,
        "type": "Transformer"
    },
    "CNNLSTM": {
        "class": CNNLSTM,
        "type": "CNNLSTM"
    },
    "BasicLSTM": {
        "class": BasicLSTM,
        "type": "LSTM"
    },
    "BidirectionalLSTM": {
        "class": BidirectionalLSTM,
        "type": "LSTM"
    },
    "ResidualLSTM": {
        "class": ResidualLSTM,
        "type": "LSTM"
    },
    "AttentionLSTM": {
        "class": AttentionLSTM,
        "type": "LSTM"
    }
}

###############################################################################
# 4. Outer loop: iterate over sensor input configurations
###############################################################################
final_results = {}

for input_mode in input_configs:
    print("\n========================================")
    print("Processing sensor input configuration:", input_mode)
    
    # Create sensor-specific log and figure directories
    sensor_log_dir = os.path.join(LOGS_DIR, input_mode)
    sensor_fig_dir = os.path.join(FIG_TRAIN_DIR, input_mode)
    os.makedirs(sensor_log_dir, exist_ok=True)
    os.makedirs(sensor_fig_dir, exist_ok=True)
    
    # Log file for current sensor config
    sensor_epoch_log_file = os.path.join(sensor_log_dir, f"loss_summary_{input_mode}.txt")
    with open(sensor_epoch_log_file, "w") as f:
        f.write(f"Loss summary for input config: {input_mode}\n\n")
    
    # Load datasets – scaling is now handled internally by the EMG_dataset
    print("Loading Train Dataset")
    train_dataset = EMG_dataset(train_index, lag=lag, n_ahead=n_ahead,
                                input_sensor=input_mode, target_sensor="emg",
                                randomize_legs=False)
    print(f"Train dataset length: {len(train_dataset)}")

    print("Loading Val Dataset")
    val_dataset = EMG_dataset(val_index, lag=lag, n_ahead=n_ahead,
                              input_sensor=input_mode, target_sensor="emg",
                              randomize_legs=False)
    print(f"Val dataset length: {len(val_dataset)}")
    
    print("Loading Test Dataset")
    test_dataset = EMG_dataset(test_index, lag=lag, n_ahead=n_ahead,
                               input_sensor=input_mode, target_sensor="emg",
                               randomize_legs=False)
    print(f"Test dataset length: {len(test_dataset)}")
    
    # Create DataLoaders (the dataset returns already-scaled inputs)
    trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    validLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    testLoader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    ###############################################################################
    # 5. Inner loop: Iterate over each model variant for the current sensor input
    ###############################################################################
    for variant_name, variant_info in model_variants.items():
        print("\n----------------------------------------")
        print(f"Training model variant: {variant_name} with sensor input: {input_mode}")
        
        # Create variant-specific log file and figure directory
        variant_log_file = os.path.join(sensor_log_dir, f"loss_summary_{variant_name}.txt")
        with open(variant_log_file, "w") as f:
            f.write(f"Loss summary for {variant_name} with input config: {input_mode}\n\n")
        variant_fig_dir = os.path.join(sensor_fig_dir, variant_name)
        os.makedirs(variant_fig_dir, exist_ok=True)
        
        input_size = input_sizes[input_mode]
        model_class = variant_info["class"]
        
        # Initialize model using the appropriate constructor
        if variant_name == "TimeSeriesTransformer":
            # Transformer constructor: (input_size, num_heads, num_layers, hidden_dim, output_size, dropout, n_ahead)
            model = model_class(input_size, num_heads, transformer_layers, transformer_hidden_dim,
                                output_size, dropout, n_ahead).to(device)
        elif variant_name == "CNNLSTM":
            # CNNLSTM constructor: (input_channels, cnn_hidden_dim, lstm_hidden_dim, lstm_num_layers, output_size)
            cnn_hidden_dim = 64
            lstm_hidden_dim = hidden_size  # you may choose to use the same hidden_size variable or a different one
            lstm_num_layers = 2  # you can adjust this parameter as needed
            model = model_class(input_size, cnn_hidden_dim, lstm_hidden_dim, lstm_num_layers, output_size).to(device)
        else:
            # LSTM-based model constructor: (input_size, hidden_size, num_layers, output_size, n_ahead)
            model = model_class(input_size, hidden_size, num_layers, output_size, n_ahead).to(device)
        
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        
        trainer = ModelTrainer(
            model=model, 
            loss=loss, 
            optimizer=optimizer, 
            accuracy=None, 
            model_type="Regression",
            model_name=variant_name,
            input_mode=input_mode,
            device=device, 
            noPrint=False, 
            flatten_output=False,
            testloader=testLoader,
            fig_dir=FIGURES_DIR
        )
        trainer.input_mode = input_mode
        trainer.epoch_log_file = variant_log_file
        
        # Check for existing checkpoints and resume if available
        checkpoint_pattern = os.path.join(
            CHECKPOINT_DIR,
            f"model_{input_mode}_{variant_name}_epoch_*.pth"
        )
        checkpoint_files = glob.glob(checkpoint_pattern)
        start_epoch = 0
        if checkpoint_files:
            latest_checkpoint = max(
                checkpoint_files,
                key=lambda x: int(re.findall(r'epoch_(\d+)\.pth', x)[0])
            )
            print(f"Resuming training from checkpoint {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed from epoch {start_epoch}")
        else:
            print("No checkpoint found, starting training from scratch.")
        
        # Train the model
        t0 = datetime.now()
        trainer.fit(trainLoader, validLoader, epochs, start_epoch=start_epoch, checkpoint_dir=CHECKPOINT_DIR)
        t1 = datetime.now()
        
        # Save the final trained model with model type in the filename
        final_model_path = os.path.join(TRAINED_DIR, f"model_{input_mode}_{variant_name}.pt")
        torch.save(model.state_dict(), final_model_path)
        print(f"Trained model saved to {final_model_path}")
        
        # Evaluate on the test set
        trainer.Test_Model(testLoader)
        print(f"Model variant: {variant_name} with sensor input: {input_mode}")
        for metric_name, metric_value in trainer.Metrics.items():
            print(f"{metric_name}: {metric_value}")
        print("Training Time:", t1 - t0)
        
        # Inference & Plotting: Prediction vs. Ground Truth
        test_results = trainer.test_results
        sample_pred = test_results["preds"][0].detach().cpu().numpy()  # shape: (n_ahead, output_size)
        sample_true = test_results["targets"][0].detach().cpu().numpy()
        
        plt.figure(figsize=(10, 6))
        for ch in range(output_size):
            plt.plot(sample_true[:, ch], label=f"Sensor {ch} EMG Actual")
            plt.plot(sample_pred[:, ch], label=f"Sensor {ch} EMG Predicted", linestyle="--")
        plt.xlabel("Time Step")
        plt.ylabel("Signal Value")
        plt.title(f"Prediction vs Ground Truth for {variant_name} ({input_mode})")
        plt.legend()
        pred_fig_path = os.path.join(variant_fig_dir, "prediction_vs_ground_truth_.png")
        plt.savefig(pred_fig_path)
        plt.close()
        print(f"Saved inference plot to {pred_fig_path}")
        # Store metrics for final summary
        final_results[f"{input_mode}_{variant_name}"] = trainer.Metrics

                # -------------------------------
        # Long-Horizon Iterative Forecasting
        # -------------------------------
        # Set model to evaluation mode
        model.eval()

        # Let's assume you want to predict a long sequence (e.g., 10000 timesteps)
        desired_length = 10000
        all_preds = []
        # Assume test_full_seq is a tensor representing the entire test sequence available for comparison.
        # You need to implement get_full_sequence() in your EMG_dataset class.
        test_full_seq = test_dataset.get_full_sequence()  # shape: [total_timesteps, input_size]
        # Initialize current window with the first "lag" timesteps from the actual sequence
        current_window = test_full_seq[:lag].clone()

        while len(all_preds) < (desired_length - lag):
            # Ensure current_window has shape [1, lag, input_size]
            with torch.no_grad():
                pred = model(current_window.unsqueeze(0))  # shape: (1, n_ahead, output_size)
            pred = pred.squeeze(0)  # shape: (n_ahead, output_size)
            all_preds.append(pred)
            # Update window: drop first n_ahead and append new prediction
            current_window = torch.cat((current_window[n_ahead:], pred), dim=0)

        # Concatenate predictions along the time axis
        predictions_full = torch.cat(all_preds, dim=0)

        # Assuming you have ground truth values for comparison; adjust as needed.
        actual_full = test_full_seq[lag:lag + predictions_full.shape[0]].cpu().numpy()
        pred_full = predictions_full.cpu().numpy()

        # Plot and save long-horizon segments
        segment_length = 1000  # or any other segment length you choose
        plot_long_horizon_segments(actual_full, pred_full, segment_length, variant_fig_dir, variant_name, input_mode, output_size)


###############################################################################
# 6. Save final summary of all experiments
###############################################################################
final_summary_file = os.path.join(LOGS_DIR, "final_loss_summary.txt")
with open(final_summary_file, "w") as f:
    f.write("Summary of Experiments (Model Variants across Sensor Inputs):\n")
    for key, metrics in final_results.items():
        test_loss_val = metrics.get("Test Loss", "N/A")
        val_loss_val  = metrics.get("Validation Loss", "N/A")
        f.write(f"{key}: Test Loss: {test_loss_val}, Validation Loss: {val_loss_val}\n")
        if "Test MSE" in metrics:
            f.write("Additional Metrics:\n")
            f.write(f"  MSE: {metrics.get('Test MSE')}\n")
            f.write(f"  RMSE: {metrics.get('Test RMSE')}\n")
            f.write(f"  MAE: {metrics.get('Test MAE')}\n")
            f.write(f"  R2: {metrics.get('Test R2')}\n")
            f.write(f"  Pearson: {metrics.get('Test Pearson')}\n")
        if "Per Action Metrics" in metrics:
            f.write("Per Action Metrics:\n")
            for act, m in metrics["Per Action Metrics"].items():
                f.write(f"  Action: {act}, Metrics: {m}\n")
        f.write("\n")
print("Final summary saved to", final_summary_file)
