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
from datetime import datetime
import time
from tqdm import tqdm

# Append parent directory to locate modules (assumes a project folder structure)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your custom dataset and trainer.
# (No changes are required in your existing trainer/datasets.py as long as the CSV column names follow the expected convention.)
from utils.datasets import EMG_dataset
from utils.Trainer import Trainer  

# Import model variants. (You can add or remove models as needed.)
from models.models import (
    LSTMModel, RNNModel, GRUModel, TCNModel, 
    TimeSeriesTransformer, TemporalTransformer, Informer, NBeats, DBN,
    PatchTST, CrossFormer, DLinear
)

# Import additional model variants.
model_variants = {
    "timeseries_transformer": TimeSeriesTransformer,
    "temporal_transformer": TemporalTransformer,
    "informer": Informer,
    "dbn": DBN,
    "nbeats": NBeats,
    "lstm": LSTMModel,
    "rnn": RNNModel,
    "gru": GRUModel,
    "tcn": TCNModel,
    "patchtst": PatchTST,
    "crossformer": CrossFormer,
    "dlinear": DLinear,
}

# ----------------------------------------------------------------------------------
# Utility function to generate trial directories.
#
# The final folder structure will be:
#   BASE_TRIAL_DIR / sensor_mode / model_name / n_ahead / loss_type
#
# For example: trials/all/lstm/1/mse/figures
# ----------------------------------------------------------------------------------
def get_trial_dir(model_name, loss_type, sensor_mode, n_ahead_val=None):
    if n_ahead_val is not None:
        trial_dir = os.path.join(BASE_TRIAL_DIR, sensor_mode, model_name, str(n_ahead_val), loss_type)
    else:
        trial_dir = os.path.join(BASE_TRIAL_DIR, sensor_mode, model_name, loss_type)
    os.makedirs(trial_dir, exist_ok=True)
    return trial_dir

# ----------------------------------------------------------------------------------
# Standard training function for a given sensor mode.
#
# Parameters:
#   - model_class: the neural network class to instantiate.
#   - model_name: name string for folder naming.
#   - loss_choice: loss function choice (e.g., "mse", "huber", etc.).
#   - sensor_mode: string indicating which sensor(s) to use for input ("all", "emg", "acc", "gyro").
#   - n_ahead_val: sliding-window prediction horizon.
#   - target_sensor: the sensor name to be used for prediction (always "emg" in our experiments by default).
#
# This function creates the appropriate trial folder structure and then loads the dataset using the given sensor_mode.
# ----------------------------------------------------------------------------------
def run_training(model_class, model_name, loss_choice, sensor_mode, n_ahead_val=None, target_sensor="emg"):
    print(f"Starting training: {model_name} (n_ahead = {n_ahead_val}) using input: {sensor_mode} with {loss_choice} loss")
    # Get the number of channels based on input sensor type.
    selected_channels = input_sizes[sensor_mode]

    # Our dataset is assumed to use a single folder called "all" (i.e. one dataset that contains all sensors).
    train_index = os.path.join(base_dir, "train_index.csv")
    val_index   = os.path.join(base_dir, "val_index.csv")
    test_index  = os.path.join(base_dir, "test_index.csv")

    # Generate the trial directory.
    trial_dir = get_trial_dir(model_name, loss_choice, sensor_mode, n_ahead_val)
    MODELS_DIR = os.path.join(trial_dir, "models")
    LOGS_DIR = os.path.join(trial_dir, "logs")
    FIGURES_DIR = os.path.join(trial_dir, "figures")
    for d in [MODELS_DIR, LOGS_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)

    # Instantiate the dataset. The 'input_sensor' is set by sensor_mode.
    train_dataset = EMG_dataset(
        processed_index_csv=train_index, lag=lag, n_ahead=n_ahead_val,
        input_sensor=sensor_mode, target_sensor=target_sensor,
        base_dir=base_dir
    )
    val_dataset = EMG_dataset(
        processed_index_csv=val_index, lag=lag, n_ahead=n_ahead_val,
        input_sensor=sensor_mode, target_sensor=target_sensor,
        base_dir=base_dir
    )
    test_dataset = EMG_dataset(
        processed_index_csv=test_index, lag=lag, n_ahead=n_ahead_val,
        input_sensor=sensor_mode, target_sensor=target_sensor, 
        base_dir=base_dir
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Configure the model.
    kwargs = {}
    
    if model_name in ["lstm", "gru", "rnn", "lstmfull", "lstmautoencoder"]:
        kwargs = {
            "input_size": selected_channels,
            "hidden_size": 128,  # Increase from default 64
            "num_layers": 3,     # Increase from default 2
            "num_classes": output_size,
            "n_ahead": n_ahead_val
        }
    elif model_name in ["tcn"]:
        kwargs = {
            "input_channels": selected_channels,
            "num_channels": [64, 128, 128, 64],  # Increased size
            "kernel_size": 3,
            "dropout": 0.2,
            "num_classes": output_size,
            "n_ahead": n_ahead_val
        }
    elif model_name == "timeseries_transformer":
        kwargs = {
            "input_size": selected_channels,
            "num_classes": output_size,
            "n_ahead": n_ahead_val,
            "d_model": 128,
            "nhead": 8,
            "num_encoder_layers": 4,
            "num_decoder_layers": 4,
            "dim_feedforward": 256,
            "dropout": 0.2
        }
    elif model_name == "temporal_transformer":
        kwargs = {
            "input_size": selected_channels,
            "num_classes": output_size,
            "n_ahead": n_ahead_val,
            "d_model": 128,
            "nhead": 8,
            "num_layers": 4,  # Not num_encoder_layers
            "dim_feedforward": 256,
            "dropout": 0.2
        }
    elif model_name == "informer":
        kwargs = {
            "enc_in": selected_channels,            # Correct parameter name
            "dec_in": selected_channels,            # Correct parameter name
            "c_out": output_size,                   # Correct parameter name
            "seq_len": lag,                         # Input sequence length
            "label_len": lag // 2,                  # Decoder conditioning length
            "out_len": n_ahead_val,                 # Forecast horizon
            "d_model": 128,
            "n_heads": 8,                           # Correct parameter name
            "e_layers": 4,                          # Correct parameter name
            "d_layers": 4,                          # Correct parameter name
            "d_ff": 256,                            # Correct parameter name
            "dropout": 0.2,
            "attn": 'prob'                          # Required parameter
        }
    elif model_name == "dbn":
        flattened_input_size = lag * selected_channels
        sizes = [flattened_input_size, 128, 64]     # List of layer sizes
        kwargs = {
            "sizes": sizes,                         # Correct parameter name
            "output_dim": output_size,              # Correct parameter name
            "n_ahead": n_ahead_val                  # Correct parameter name
        }
    elif model_name == "nbeats":
        flattened_input_size = lag * selected_channels
        kwargs = {
            "input_size": flattened_input_size,
            "output_size": output_size,             # Correct parameter name
            "n_ahead": n_ahead_val,                 # Correct parameter name
            "num_blocks_per_stack": 3,              # Correct parameter name
            "stack_types": ["trend", "seasonality"] # Required parameter
        }
    elif model_name == "patchtst":
        kwargs = {
            "input_channels": selected_channels,
            "patch_size": 16,
            "d_model": 512,
            "nhead": 8,
            "num_layers": 3,
            "dim_feedforward": 2048,
            "dropout": 0.1,
            "forecast_horizon": n_ahead_val,
            "output_size": output_size
        }
    elif model_name == "crossformer":
        kwargs = {
            "input_channels": selected_channels,
            "seq_len": lag,
            "d_model": 64,
            "nhead": 8,
            "num_layers": 3,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "forecast_horizon": n_ahead_val,
            "output_size": output_size
        }
    elif model_name == "dlinear":
        kwargs = {
            "seq_len": lag,
            "forecast_horizon": n_ahead_val,
            "num_channels": selected_channels,
            "individual": False,
            "moving_avg_kernel": 25
        }
    else:  # Fallback for any other models
        kwargs = {
            "input_size": selected_channels,
            "num_classes": output_size,
            "n_ahead": n_ahead_val,
            "d_model": 128,
            "nhead": 8,
            "num_encoder_layers": 4,
            "num_decoder_layers": 4,
            "dim_feedforward": 256,
            "dropout": 0.2
        }

    model = model_class(**kwargs)
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Set the optimizer and scheduler.
    optimizer = optim.Adam(model.parameters(), lr=fast_lr, weight_decay=1e-4)  
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5, min_lr=1e-6)

    # Configure and train the model.
    trainer = Trainer(
        model=model,
        lag=lag,
        n_ahead=n_ahead_val,
        optimizer=optimizer,
        scheduler=scheduler,
        testloader=test_loader,
        fig_dir=FIGURES_DIR,
        loss=loss_choice,
        model_type=model_name,
        device=device,
        clip_grad_norm=2.0  
    )

    epoch_log_file = os.path.join(LOGS_DIR, f"loss_summary_{model_name}.txt")
    trainer.epoch_log_file = epoch_log_file

    # Checkpointing: resume if previous checkpoints exist.
    checkpoint_dir = os.path.join(trial_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    epoch_checkpoint_pattern = checkpoint_dir + "/checkpoint_epoch_*.pt"
    epoch_checkpoint_files = glob.glob(epoch_checkpoint_pattern)
    start_epoch = 1  # Default to 1 instead of 0
    if epoch_checkpoint_files:
        latest_checkpoint_file = max(epoch_checkpoint_files, key=lambda x: int(re.findall(r'epoch_(\d+)\.pt', x)[0]))
        checkpoint = torch.load(latest_checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch} for {model_name}/{loss_choice} with input {sensor_mode} ...")
    else:
        print("No epoch checkpoints found. Starting training from epoch 1.")

    remaining_epochs = default_epochs - start_epoch
    loss_curve = f'{trial_dir}_{n_ahead_val}_{loss_choice}.png'
    trainer.fit(train_loader, val_loader, epochs=remaining_epochs, patience=10, min_delta=0.0,
                checkpoint_dir=checkpoint_dir, loss_curve_path=loss_curve)
    
    test_save_path = os.path.join(trial_dir, "test_results.png")
    trainer.Test_Model(test_loader, test_save_path)
    val_plot_path = os.path.join(trial_dir, "validation_results.png")
    trainer.plot_validation_results(val_loader, val_plot_path)

    final_model_path = os.path.join(MODELS_DIR, f"model_{model_name}.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # ------------------------------
    # ONNX Exporting
    # ------------------------------
    model_for_export = model.module if hasattr(model, "module") else model
    model_for_export.eval()
    # Dummy input shape: (1, lag, selected_channels)
    dummy_input = torch.randn(1, lag, selected_channels, device=device)
    onnx_model_path = os.path.join(MODELS_DIR, f"model_{model_name}.onnx")
    torch.onnx.export(
        model_for_export,
        dummy_input,
        onnx_model_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'seq_len'},
            'output': {0: 'batch_size', 1: 'seq_len'}
        },
        opset_version=14
    )
    print(f"ONNX model exported to {onnx_model_path}")
    
    return trainer.Metrics

# Wrapper function to run training with proper error handling.
def train_model(model_name, model_cls, loss_func, sensor_mode, n_val, target_sensor):
    try:
        print(f"Starting training for: {model_name} (sensor: {sensor_mode}, n_ahead: {n_val}, loss: {loss_func})")
        run_training(model_cls, model_name, loss_func, sensor_mode, n_val, target_sensor)
        return True
    except Exception as e:
        print(f"Error training {model_name}: {str(e)}")
        return None

# ----------------------------------------------------------------------------------
# DEVICE / PATH CONFIGURATION & HYPERPARAMETERS
# ----------------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the base folder for trials.
BASE_TRIAL_DIR = "/data1/dnicho26/Thesis/AI-Assisted-Gait-Restoration-for-Disabled-Individuals/trials-acc"
os.makedirs(BASE_TRIAL_DIR, exist_ok=True)

# Define the base directory for your dataset.
# In this example, we assume that /data1/dnicho26/EMG_DATASET/final-data-test/all/ contains train.csv, val.csv, and test.csv.
base_dir = "/data1/dnicho26/EMG_DATASET/final-data/"

lag = 30         # Length of the sliding window.
n_ahead = 10      # Default prediction horizon (this value will be passed as n_ahead_val).
batch_size = 12
default_epochs = 300
fast_lr = 1e-4
final_lr = 7e-4
output_size = 9
target_sensor = "acc"  

# Define input sizes (number of channels) for each sensor mode.
input_sizes = {"all": 21, "emg": 3, "acc": 9, "gyro": 9}

# Loss functions (choose the one you wish to use in each trial).
LOSS_TYPES = ["huber", "custom"]

# ----------------------------------------------------------------------------------
# Main Execution Block
#
# This block loops over sensor modes so that training runs are saved to:
#   trials/{sensor_mode}/{model_name}/{n_ahead}/{loss_type}/...
#
# For example, one trial might be:
#   trials/all/lstm/1/mse/figures
#   and another:
#   trials/gyro/lstm/1/mse/figures
#
# Modify the loops below to run the desired experiments.
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Starting training on device: {device}")
    
    # List of sensor modes to run experiments.
    sensor_modes = ["acc","all"]    # Define a list of prediction horizons (n_ahead values) to experiment with.
    n_ahead_values = [5,10,15,20] 
    # Create a list of all combinations of experiments
    experiments = []
    for sensor_mode in sensor_modes:
        for n_val in n_ahead_values:
            for loss_func in LOSS_TYPES:
                for model_name, model_cls in model_variants.items():
                    experiments.append((model_name, model_cls, loss_func, sensor_mode, n_val, target_sensor))

    print(f"Total experiments: {len(experiments)}")
    
    # Run experiments sequentially
    for model_name, model_cls, loss_func, sensor_mode, n_val, target_sensor in experiments:
        try:
            train_model(model_name, model_cls, loss_func, sensor_mode, n_val, target_sensor)
            print(f"Completed training for {model_name} with {loss_func} loss")
        except Exception as e:
            print(f"Error during training: {str(e)}")

    print("\nAll training runs completed!")
