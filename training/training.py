import os
import sys
import glob
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from datetime import datetime
import concurrent.futures  # <-- For threading

# Append parent directory to locate modules (assumes a project folder structure)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.datasets import EMG_dataset
from utils.Trainer import Trainer  
from models.models import (
    LSTMModel, RNNModel, GRUModel, TCNModel, HybridTransformerLSTM,
    LSTMFullSequence, LSTMAutoencoder,
    TimeSeriesTransformer, TemporalTransformer, Informer, NBeats, DBN
)

# ----------------------------------------------------------------------------------
# Utility functions to generate trial directories.
# ----------------------------------------------------------------------------------
def get_trial_dir(model_name, loss_type, dataset_name, n_ahead_val=None):
    if n_ahead_val is not None:
        trial_dir = os.path.join(BASE_TRIAL_DIR, dataset_name, model_name, str(n_ahead_val), loss_type)
    else:
        trial_dir = os.path.join(BASE_TRIAL_DIR, dataset_name, model_name, loss_type)
    os.makedirs(trial_dir, exist_ok=True)
    return trial_dir

def get_sensor_trial_dir(model_name, loss_type, dataset_name, sensor_pair, sensor_mode, n_ahead_val=None):
    if n_ahead_val is not None:
        trial_dir = os.path.join(BASE_SENSOR_TRIAL_DIR, dataset_name, model_name, str(n_ahead_val), loss_type, sensor_mode, f"sensor_{sensor_pair}")
    else:
        trial_dir = os.path.join(BASE_SENSOR_TRIAL_DIR, dataset_name, model_name, loss_type, sensor_mode, f"sensor_{sensor_pair}")
    os.makedirs(trial_dir, exist_ok=True)
    return trial_dir

# ----------------------------------------------------------------------------------
# Standard training function updated to use trainer.fit and checkpoint resuming
# ----------------------------------------------------------------------------------
def run_training(model_class, model_name, loss_choice, dataset_name, n_ahead_val=None):
    print(f"Starting standard training: {model_name} (n_ahead = {n_ahead_val}) on {dataset_name} with {loss_choice} loss")
    n_ahead_run = n_ahead_val if n_ahead_val is not None else n_ahead

    base_dir_dataset = os.path.join(base_dir, dataset_name)
    train_index = os.path.join(base_dir_dataset, "train.csv")
    val_index   = os.path.join(base_dir_dataset, "val.csv")
    test_index  = os.path.join(base_dir_dataset, "test.csv")

    trial_dir = get_trial_dir(model_name, loss_choice, dataset_name, n_ahead_val)
    MODELS_DIR = os.path.join(trial_dir, "models")
    LOGS_DIR = os.path.join(trial_dir, "logs")
    FIGURES_DIR = os.path.join(trial_dir, "figures")
    for d in [MODELS_DIR, LOGS_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)
        
    train_dataset = EMG_dataset(
        train_index, lag=lag, n_ahead=n_ahead_run,
        input_sensor=input_mode, target_sensor="emg", randomize_legs=False,
        base_dir=base_dir
    )
    test_dataset = EMG_dataset(
        test_index, lag=lag, n_ahead=n_ahead_run,
        input_sensor=input_mode, target_sensor="emg", randomize_legs=False,
        base_dir=base_dir
    )
    val_dataset = EMG_dataset(
        val_index, lag=lag, n_ahead=n_ahead_run,
        input_sensor=input_mode, target_sensor="emg", randomize_legs=False,
        base_dir=base_dir
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    distribution_fig = os.path.join(FIGURES_DIR, f"{dataset_name}_distribution.png")
    
    # Instantiate model based on type
    if model_name == "tcn":
        model = model_class(input_channels=input_sizes[input_mode],
                            num_classes=output_size, n_ahead=n_ahead_run).to(device)
    elif model_name in {"timeseries_transformer", "temporal_transformer", "informer"}:        
        model = model_class(input_size=input_sizes[input_mode], num_classes=output_size, n_ahead=n_ahead_run).to(device)
    elif model_name == "nbeats":
        flattened_input_size = lag * input_sizes[input_mode]
        model = model_class(input_size=flattened_input_size, num_stacks=3, num_blocks_per_stack=3, num_layers=4,
                            hidden_size=256, output_size=output_size, n_ahead=n_ahead_run).to(device)
    elif model_name == "dbn":
        flattened_input_size = lag * input_sizes[input_mode]  # e.g. 30 * 3 = 90
        sizes = [flattened_input_size, 128, 128]
        model = model_class(sizes=sizes, output_dim=output_size, n_ahead=n_ahead_run).to(device)
        print("Starting DBN pretraining...")
        model.pretrain(train_loader, num_epochs=10, batch_size=batch_size, verbose=True)
    else:
        model = model_class(input_size=input_sizes[input_mode], hidden_size=256, num_layers=5,
                            num_classes=output_size, n_ahead=n_ahead_run).to(device)
    train_dataset.plot_distribution(fig_path=distribution_fig)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=fast_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=lambda epoch: (epoch+1)/5 if epoch < 5 else final_lr/fast_lr)

    epoch_log_file = os.path.join(LOGS_DIR, f"loss_summary_{input_mode}_{model_name}.txt")

    # Instantiate the Trainer (using early stopping via fit)
    trainer = Trainer(model=model, loss=loss_choice, optimizer=optimizer, scheduler=scheduler,
                      testloader=test_loader, fig_dir=FIGURES_DIR, model_type=model_name, device=device,
                      lag=lag, n_ahead=n_ahead_run)
    trainer.epoch_log_file = epoch_log_file

    # Define checkpoint directory and patterns
    checkpoint_dir = os.path.join(trial_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    epoch_checkpoint_pattern = os.path.join(checkpoint_dir, f"model_{input_mode}_{model_name}_epoch_*.pth")
    epoch_checkpoint_files = glob.glob(epoch_checkpoint_pattern)

    # Define checkpoint directory and patterns
    checkpoint_dir = os.path.join(trial_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Define an epoch checkpoint prefix and best checkpoint path
    epoch_checkpoint_prefix = os.path.join(checkpoint_dir, f"model_{input_mode}_{model_name}")
    epoch_checkpoint_pattern = epoch_checkpoint_prefix + "_epoch_*.pth"
    epoch_checkpoint_files = glob.glob(epoch_checkpoint_pattern)
    start_epoch = 0
    if epoch_checkpoint_files:
        latest_checkpoint_file = max(epoch_checkpoint_files, key=lambda x: int(re.findall(r'epoch_(\d+)\.pth', x)[0]))
        checkpoint = torch.load(latest_checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch} for {model_name}/{loss_choice} on {dataset_name} ...")
    else:
        print("No epoch checkpoints found. Starting training from epoch 0.")


    remaining_epochs = default_epochs - start_epoch

    # Define a best checkpoint path for early stopping
    best_checkpoint_path = os.path.join(checkpoint_dir, f"best_checkpoint_{input_mode}_{model_name}.pth")
    # Call the fit method (which saves per-epoch and best checkpoints)
    loss_curve=f'{trial_dir}_{n_ahead_val}_{loss_choice}.png'

    trainer.fit(train_loader, val_loader, epochs=remaining_epochs, patience=5, min_delta=0.0,
                best_checkpoint_path=best_checkpoint_path,loss_curve_path=loss_curve)

    # After training, load the best checkpoint saved by early stopping
    best_checkpoint = torch.load(best_checkpoint_path)
    model.load_state_dict(best_checkpoint)
    
    # Evaluate on the test set and plot validation results
    trainer.Test_Model(test_loader)
    val_plot_path = os.path.join(trial_dir, "validation_results.png")
    trainer.plot_validation_results(val_loader, val_plot_path)

    # Optionally, save the final model
    final_model_path = os.path.join(MODELS_DIR, f"model_{input_mode}_{model_name}.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    return trainer.Metrics

# ----------------------------------------------------------------------------------
# Sensor pair training function (similar modifications as above)
# ----------------------------------------------------------------------------------
def run_training_sensor_pair(model_class, model_name, loss_choice, dataset_name, sensor_pair, sensor_mode, n_ahead_val=None):
    print(f"Starting sensor pair training: {model_name} (n_ahead = {n_ahead_val}) on {dataset_name} with {loss_choice} loss")
    n_ahead_run = n_ahead_val if n_ahead_val is not None else n_ahead

    trial_dir = get_sensor_trial_dir(model_name, loss_choice, dataset_name, sensor_pair, sensor_mode, n_ahead_val)
    MODELS_DIR = os.path.join(trial_dir, "models")
    LOGS_DIR = os.path.join(trial_dir, "logs")
    FIGURES_DIR = os.path.join(trial_dir, "figures")
    for d in [MODELS_DIR, LOGS_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)
        
    base_dir_dataset = os.path.join(base_dir, dataset_name)
    train_index = os.path.join(base_dir_dataset, "train.csv")
    val_index   = os.path.join(base_dir_dataset, "val.csv")
    test_index  = os.path.join(base_dir_dataset, "test.csv")
    
    train_dataset = EMG_dataset(
        train_index, lag=lag, n_ahead=n_ahead_run,
        input_sensor=sensor_mode, target_sensor="emg", randomize_legs=False,
        sensor_pair=sensor_pair,
        base_dir=base_dir
    )
    test_dataset = EMG_dataset(
        test_index, lag=lag, n_ahead=n_ahead_run,
        input_sensor=sensor_mode, target_sensor="emg", randomize_legs=False,
        sensor_pair=sensor_pair,
        base_dir=base_dir
    )
    val_dataset = EMG_dataset(
        val_index, lag=lag, n_ahead=n_ahead_run,
        input_sensor=sensor_mode, target_sensor="emg", randomize_legs=False,
        sensor_pair=sensor_pair,
        base_dir=base_dir
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    distribution_fig = os.path.join(FIGURES_DIR, f"{dataset_name}_sensor_pair_{sensor_pair}_distribution.png")
    train_dataset.plot_distribution(distribution_fig)
    
    if model_name == "tcn":
        model = model_class(input_channels=1, num_classes=output_size, n_ahead=n_ahead_run).to(device)
    elif model_name in {"timeseries_transformer", "temporal_transformer", "informer"}:
        model = model_class(input_size=1, num_classes=output_size, n_ahead=n_ahead_run).to(device)
    elif model_name == "nbeats":
        model = model_class(input_size=lag*1, num_stacks=3, num_blocks_per_stack=3,
                            num_layers=4, hidden_size=256, output_size=1, n_ahead=n_ahead_run).to(device)
    elif model_name == "dbn":
        flattened_input_size = lag * 1  # since sensor pairs have one channel
        sizes = [flattened_input_size, 128, 128]
        model = model_class(sizes=sizes, output_dim=1, n_ahead=n_ahead_run).to(device)
        print("Starting DBN pretraining for sensor pair...")
        model.pretrain(train_loader, num_epochs=10, batch_size=batch_size, verbose=True)
    else:
        model = model_class(input_size=1, hidden_size=256, num_layers=5, num_classes=1, n_ahead=n_ahead_run).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=fast_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=lambda epoch: (epoch+1)/5 if epoch < 5 else final_lr/fast_lr)

    epoch_log_file = os.path.join(LOGS_DIR, f"loss_summary_sensor_{sensor_pair}_{model_name}.txt")
    trainer = Trainer(model=model, loss=loss_choice, optimizer=optimizer, scheduler=scheduler,
                      testloader=test_loader, fig_dir=FIGURES_DIR, model_type=model_name, device=device,
                      lag=lag, n_ahead=n_ahead_run)
    trainer.epoch_log_file = epoch_log_file

    # Define checkpoint directory and patterns
    checkpoint_dir = os.path.join(trial_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Define an epoch checkpoint prefix and best checkpoint path
    epoch_checkpoint_prefix = os.path.join(checkpoint_dir, f"model")
    epoch_checkpoint_pattern = epoch_checkpoint_prefix + "_checkpoint_epoch_*.pth"
    epoch_checkpoint_files = glob.glob(epoch_checkpoint_pattern)
    start_epoch = 0
    if epoch_checkpoint_files:
        latest_checkpoint_file = max(epoch_checkpoint_files, key=lambda x: int(re.findall(r'epoch_(\d+)\.pth', x)[0]))
        checkpoint = torch.load(latest_checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch} for {model_name}/{loss_choice} on {dataset_name} ...")
    else:
        print("No epoch checkpoints found. Starting training from epoch 0.")

    remaining_epochs = default_epochs - start_epoch
    loss_curve=f'{trial_dir}_{n_ahead_val}_{loss_choice}.png'
    trainer.fit(train_loader, val_loader, epochs=remaining_epochs, patience=5, min_delta=0.0,
                checkpoint_dir=checkpoint_dir,loss_curve_path=loss_curve)
    test_save_path = os.path.join(trial_dir, "test_results.png")
    trainer.Test_Model(test_loader,test_save_path)
    val_plot_path = os.path.join(trial_dir, "validation_results.png")
    trainer.plot_validation_results(val_loader, val_plot_path)

    final_model_path = os.path.join(MODELS_DIR, f"model_sensor_{sensor_pair}_{model_name}.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    return trainer.Metrics

# ----------------------------------------------------------------------------------
# DEVICE / PATH CONFIG
# ----------------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
BASE_TRIAL_DIR = "/data1/dnicho26/Thesis/AI-Assisted-Gait-Restoration-for-Disabled-Individuals/trials"
os.makedirs(BASE_TRIAL_DIR, exist_ok=True)
BASE_SENSOR_TRIAL_DIR = os.path.join(BASE_TRIAL_DIR, "sensor-pairs")
os.makedirs(BASE_SENSOR_TRIAL_DIR, exist_ok=True)

# ----------------------------------------------------------------------------------
# DATASET CONFIG & HYPERPARAMETERS
# ----------------------------------------------------------------------------------
base_dir = "/data1/dnicho26/EMG_DATASET/final-data/"
lag = 30         # try 100
n_ahead = 4      # default (will be replaced by the n_ahead_val in each trial)
batch_size = 128
default_epochs = 300
fast_lr = 1e-4
final_lr = 7e-4
output_size = 3

# For standard training, we use an input_mode and input_sizes dictionary.
input_mode = "emg"
input_sizes = {"emg": 3}

# Define loss functions.
LOSS_TYPES = ["custom", "huber", "mse", "smoothl1"]

# Define model variants.
model_variants = {
    "nbeats": NBeats,
    "timeseries_transformer": TimeSeriesTransformer,
    "temporal_transformer": TemporalTransformer,
    "informer": Informer,
    "dbn": DBN,
    "hybrid": HybridTransformerLSTM,
    "lstm": LSTMModel,
    "lstmauto": LSTMAutoencoder,
    "lstmfull": LSTMFullSequence,
    "rnn": RNNModel,
    "gru": GRUModel,
    "tcn": TCNModel,
}

# ----------------------------------------------------------------------------------
# Dataset-Specific Sensor Mode Configurations
# ----------------------------------------------------------------------------------
ds_sensor_modes = {
    "DS1": [("all", 21)],
    "DS2": [("emg", 3)],
    "DS3": [("acc_emg", 9)],
    "DS4": [("gyro_emg", 9)]
}

# ----------------------------------------------------------------------------------
# Main execution block
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Starting training on device: {device}")
    dataset_names = ["DS1", "DS2", "DS3", "DS4"]
    
    # Define the n_ahead values to run for every model.
    n_ahead_values = [1, 2, 3, 4]
    
    # Standard training runs.
    for loss_func in LOSS_TYPES:
        # Sensor pair training runs.
        for ds in dataset_names:
            for sensor_mode, available_channels in ds_sensor_modes[ds]:
                for sensor in range(available_channels):
                    for model_name, model_cls in model_variants.items():
                        for n_val in n_ahead_values:
                            run_training_sensor_pair(model_cls, model_name, loss_func, ds, sensor, sensor_mode, n_val)

        for ds in dataset_names:
            for model_name, model_cls in model_variants.items():
                for n_val in n_ahead_values:
                    run_training(model_cls, model_name, loss_func, ds, n_val)
    
    print("All single sensor runs completed!")
    print("All training runs completed!")
