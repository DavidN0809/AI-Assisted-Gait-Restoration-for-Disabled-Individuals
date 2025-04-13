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
from utils.datasets import EMG_dataset_window_norm
from utils.Trainer import Trainer  

# Import model variants. (You can add or remove models as needed.)
from models.models import (
    LSTMModel, RNNModel, GRUModel, TCNModel,
    TimeSeriesTransformer, TemporalTransformer, Informer, NBeats, DBN,
    PatchTST, CrossFormer, DLinear
)

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
    train_dataset = EMG_dataset_window_norm(
        input_size=256,
        processed_index_csv=train_index, lag=lag, n_ahead=n_ahead_val,
        input_sensor=sensor_mode, target_sensor=target_sensor,
        base_dir=base_dir
    )
    val_dataset = EMG_dataset_window_norm(
        input_size=256,
        processed_index_csv=val_index, lag=lag, n_ahead=n_ahead_val,
        input_sensor=sensor_mode, target_sensor=target_sensor,
        base_dir=base_dir
    )
    test_dataset = EMG_dataset_window_norm(
        input_size=256,
        processed_index_csv=test_index, lag=lag, n_ahead=n_ahead_val,
        input_sensor=sensor_mode, target_sensor=target_sensor, 
        base_dir=base_dir
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Configure the model based on the model name.
    if model_name == "tcn":
        model = model_class(
            input_channels=selected_channels, 
            num_classes=output_size, 
            n_ahead=n_ahead_val
        ).to(device)
    elif model_name == "timeseries_transformer":
        model = model_class(
            input_size=selected_channels, 
            output_size=output_size, 
            n_ahead=n_ahead_val
        ).to(device)
    elif model_name == "temporal_transformer":
        model = model_class(
            input_size=selected_channels, 
            num_classes=output_size, 
            n_ahead=n_ahead_val
        ).to(device)
    elif model_name == "nbeats":
        flattened_input_size = lag * selected_channels
        model = model_class(
            input_size=flattened_input_size,
            output_size=output_size,
            stack_types=["trend", "seasonality"],
            num_blocks_per_stack=3,
            n_ahead=n_ahead_val
        ).to(device)
    elif model_name == "informer":
        label_len = lag // 2
        model = model_class(
            enc_in=selected_channels,
            dec_in=selected_channels,
            c_out=output_size,
            seq_len=lag,
            label_len=label_len,
            out_len=n_ahead_val,
            activation="relu",
            output_attention=False
        ).to(device)
    elif model_name == "dbn":
        flattened_input_size = lag * selected_channels
        sizes = [flattened_input_size, 128, 64]
        model = model_class(
            sizes=sizes, 
            output_dim=output_size, 
            n_ahead=n_ahead_val
        ).to(device)
        print("Starting DBN pretraining...")
        model.pretrain(train_loader, num_epochs=5, batch_size=batch_size, verbose=True)
    else:
        # Default to recurrent models (LSTM, RNN, GRU, etc.)
        model = model_class(
            input_size=selected_channels, 
            hidden_size=128, 
            num_layers=3,
            num_classes=output_size, 
            n_ahead=n_ahead_val
        ).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=fast_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=lambda epoch: (epoch+1)/5 if epoch < 5 else final_lr/fast_lr)

    epoch_log_file = os.path.join(LOGS_DIR, f"loss_summary_{model_name}.txt")
    trainer = Trainer(model=model, loss=loss_choice, optimizer=optimizer, scheduler=scheduler,
                     testloader=test_loader, fig_dir=FIGURES_DIR, model_type=model_name, device=device,
                      lag=lag, n_ahead=n_ahead_val)
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

def train_model(model_name, model_cls, loss_func, sensor_mode, n_val, target_sensor):
    """Wrapper function to run training with proper error handling."""
    try:
        print(f"\nStarting training for: {model_name} (sensor: {sensor_mode}, n_ahead: {n_val}, loss: {loss_func})")
        return run_training(model_cls, model_name, loss_func, sensor_mode, n_val, target_sensor)
    except Exception as e:
        print(f"Error training {model_name}: {str(e)}")
        return None

# ----------------------------------------------------------------------------------
# DEVICE / PATH CONFIGURATION & HYPERPARAMETERS
# ----------------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the base folder for trials.
BASE_TRIAL_DIR = "/data1/dnicho26/Thesis/AI-Assisted-Gait-Restoration-for-Disabled-Individuals/trials-not-norm"
os.makedirs(BASE_TRIAL_DIR, exist_ok=True)

# Define the base directory for your dataset.
# In this example, we assume that /data1/dnicho26/EMG_DATASET/final-data-test/all/ contains train.csv, val.csv, and test.csv.
base_dir = "/data1/dnicho26/EMG_DATASET/final-data-not-norm/"

lag = 30         # Length of the sliding window.
n_ahead = 10      # Default prediction horizon (this value will be passed as n_ahead_val).
batch_size = 12
default_epochs = 300
fast_lr = 1e-4
final_lr = 7e-4
output_size = 3
target_sensor = "emg"  

# Define input sizes (number of channels) for each sensor mode.
input_sizes = {"all": 21, "emg": 3, "acc": 9, "gyro": 9}

# Loss functions (choose the one you wish to use in each trial).
LOSS_TYPES = ["huber", "custom"]

# Define model variants.
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
    sensor_modes = ["emg","all"]    # Define a list of prediction horizons (n_ahead values) to experiment with.
    n_ahead_values = [5,10,15,20]      
    # Choose a loss function and model variant for demonstration.
    loss_func = "mse"  # Can be "huber", "mse", "smoothl1", or "custom"

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
