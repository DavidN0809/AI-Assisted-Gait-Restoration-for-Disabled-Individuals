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

# Append parent directory to locate modules (assumes a project folder structure)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.datasets import EMG_dataset
from utils.Trainer import Trainer  
from models.models import (
    LSTMModel, RNNModel, GRUModel, TCNModel, HybridTransformerLSTM,
    LSTMFullSequence, LSTMAutoregressive, LSTMDecoder,
    TimeSeriesTransformer, TemporalTransformer, Informer, NBeats
)

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
lag = 30 # try 100
n_ahead = 2 # try 50
batch_size = 128
default_epochs = 150
fast_lr = 1e-4
final_lr = 7e-4
output_size = 3

# For standard training, we use an input_mode and input_sizes dictionary.
input_mode = "emg"
input_sizes = {"emg": 3}

# Define loss functions.
LOSS_TYPES = ["huber", "mse", "smoothl1"]

# Define model variants as a dictionary.
model_variants = {
    "timeseries_transformer": TimeSeriesTransformer,
    "temporal_transformer": TemporalTransformer,
    "lstmauto": LSTMAutoregressive,
    "lstmfull": LSTMFullSequence,
    "lstmdecoder":LSTMDecoder,
    "tcn": TCNModel,
    "hyrbid": HybridTransformerLSTM,
    "lstm": LSTMModel,
    "rnn": RNNModel,
    "gru": GRUModel,
    "informer": Informer,
    "nbeats": NBeats,
}

# ----------------------------------------------------------------------------------
# Dataset-Specific Sensor Mode Configurations
# ----------------------------------------------------------------------------------
# Each dataset maps to a tuple: (sensor_mode, number_of_channels_per_leg)
# DS1: All sensors are available (emg, acc, gyro) → 3 channels.
# DS2: Only emg is available → 1 channel.
# DS3: Both acc and emg → 2 channels.
# DS4: Both gyro and emg → 2 channels.
ds_sensor_modes = {
    "DS1": [("all", 21)],
    "DS2": [("emg", 3)],
    "DS3": [("acc_emg", 9)],
    "DS4": [("gyro_emg", 9)]
}

# ----------------------------------------------------------------------------------
# Utility functions to generate trial directories.
# ----------------------------------------------------------------------------------
def get_trial_dir(model_name, loss_type, dataset_name):
    trial_dir = os.path.join(BASE_TRIAL_DIR, dataset_name, model_name, loss_type)
    os.makedirs(trial_dir, exist_ok=True)
    return trial_dir

def get_sensor_trial_dir(model_name, loss_type, dataset_name, sensor_pair, sensor_mode):
    # New trial directory for sensor pair training.
    trial_dir = os.path.join(BASE_SENSOR_TRIAL_DIR, dataset_name, model_name, loss_type, f"{sensor_mode}", f"sensor_{sensor_pair}")
    os.makedirs(trial_dir, exist_ok=True)
    return trial_dir

# ----------------------------------------------------------------------------------
# Standard training function
# ----------------------------------------------------------------------------------
def run_training(model_class, model_name, loss_choice, dataset_name):
    base_dir_dataset = os.path.join(base_dir, dataset_name)
    train_index = os.path.join(base_dir_dataset, "train.csv")
    val_index   = os.path.join(base_dir_dataset, "val.csv")
    test_index  = os.path.join(base_dir_dataset, "test.csv")
    print(f"Length of each index {len(train_index)}")
    print(f"Length of each index {len(val_index)}")
    print(f"Length of each index {len(test_index)}")

    trial_dir = get_trial_dir(model_name, loss_choice, dataset_name)
    MODELS_DIR = os.path.join(trial_dir, "models")
    LOGS_DIR = os.path.join(trial_dir, "logs")
    FIGURES_DIR = os.path.join(trial_dir, "figures")
    for d in [MODELS_DIR, LOGS_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)
        
    print(f"Loading Datasets for {dataset_name}")
    train_dataset = EMG_dataset(
        train_index, lag=lag, n_ahead=n_ahead,
        input_sensor=input_mode, target_sensor="emg", randomize_legs=False,
        base_dir=base_dir
    )
    test_dataset = EMG_dataset(
        test_index, lag=lag, n_ahead=n_ahead,
        input_sensor=input_mode, target_sensor="emg", randomize_legs=False,
        base_dir=base_dir
    )
    val_dataset = EMG_dataset(
        val_index, lag=lag, n_ahead=n_ahead,
        input_sensor=input_mode, target_sensor="emg", randomize_legs=False,
        base_dir=base_dir
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    distribution_fig = os.path.join(FIGURES_DIR, f"{dataset_name}_distribution.png")
    
    # Instantiate model using the standard input size.
    if model_name in {"timeseries_transformer", "temporal_transformer", "informer"}:        
        model = model_class(input_size=input_sizes[input_mode], num_classes=output_size, n_ahead=n_ahead).to(device)
    elif model_name == "nbeats":
        model = model_class(input_size=input_sizes[input_mode], num_stacks=3, num_blocks_per_stack=3, num_layers=4, hidden_size=256, output_size=output_size, n_ahead=n_ahead).to(device)
    else:
        model = model_class(input_size=input_sizes[input_mode], hidden_size=256, num_layers=5, num_classes=output_size, n_ahead=n_ahead).to(device)
    train_dataset.plot_distribution(fig_path=distribution_fig)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=fast_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch+1)/5 if epoch < 5 else final_lr/fast_lr)

    epoch_log_file = os.path.join(LOGS_DIR, f"loss_summary_{input_mode}_{model_name}.txt")
    trainer = Trainer(model=model, loss=loss_choice, optimizer=optimizer, scheduler=scheduler,
                      testloader=test_loader, fig_dir=FIGURES_DIR, model_type=model_name, device=device,
                      use_variation_penalty=True, alpha=1.0, var_threshold=0.01, lag=lag, n_ahead=n_ahead)
    trainer.epoch_log_file = epoch_log_file

    checkpoint_dir = os.path.join(trial_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_pattern = os.path.join(checkpoint_dir, f"model_{input_mode}_{model_name}_epoch_*.pth")
    checkpoint_files = glob.glob(checkpoint_pattern)
    start_epoch = 0
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(re.findall(r'epoch_(\d+)\.pth', x)[0]))
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch} for {model_name} / {loss_choice} on {dataset_name} ...")

    for epoch in range(start_epoch, default_epochs):
        trainer.Training_Loop(train_loader)
        trainer.Validation_Loop(val_loader)
        trainer.step_scheduler()

        print(f"\nEpoch {epoch+1}/{default_epochs} for {model_name} ({loss_choice}) on {dataset_name}")
        print("  Train Loss:", trainer.Metrics["Training Loss"][-1])
        print("  Valid Loss:", trainer.Metrics["Validation Loss"][-1])

        checkpoint_path = os.path.join(checkpoint_dir, f"model_{input_mode}_{model_name}_epoch_{epoch+1}.pth")
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
        print(f"  Checkpoint saved to {checkpoint_path}")

        trainer.save_first_10_windows(train_loader, epoch+1)

    final_model_path = os.path.join(MODELS_DIR, f"model_{input_mode}_{model_name}.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    trainer.Test_Model(test_loader)
    val_plot_path = os.path.join(trial_dir, "validation_results.png")
    trainer.plot_validation_results(val_loader, val_plot_path)

    test_results = trainer.test_results
    sample_idx = 0
    test_batch = next(iter(test_loader))
    X_sample, Y_sample, *_ = test_batch
    X_sample = X_sample[sample_idx].detach().cpu().numpy()
    Y_sample = Y_sample[sample_idx].detach().cpu().numpy()
    Y_pred = test_results["preds"][sample_idx].detach().cpu().numpy()
    channel = 0
    combined_true = np.concatenate([X_sample[:, channel], Y_sample[:, channel]])
    combined_pred = np.concatenate([X_sample[:, channel], Y_pred[:, channel]])

    plt.figure(figsize=(10, 6))
    plt.plot(combined_true, label="Ground Truth", marker="o")
    plt.plot(combined_pred, label="Prediction", marker="x", linestyle="--")
    plt.axvline(x=len(X_sample)-0.5, color="black", linestyle="--", label="Forecast Boundary")
    plt.xlabel("Time Step")
    plt.ylabel("Signal Value")
    plt.title(f"Combined Prediction vs Ground Truth ({model_name} / {loss_choice}) on {dataset_name}")
    plt.legend()
    final_fig_path = os.path.join(trial_dir, "combined_prediction.png")
    plt.savefig(final_fig_path)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(trainer.Metrics["Training Loss"], label="Training Loss")
    plt.plot(trainer.Metrics["Validation Loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve ({model_name} / {loss_choice}) on {dataset_name}")
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(trial_dir, "loss_curve.png")
    plt.savefig(loss_plot_path)
    plt.close()

    return trainer.Metrics

# ----------------------------------------------------------------------------------
# Sensor pair training function (with sensor mode parameter)
# ----------------------------------------------------------------------------------
def run_training_sensor_pair(model_class, model_name, loss_choice, dataset_name, sensor_pair, sensor_mode):
    # For sensor pair training, we select one channel per leg.
    trial_dir = get_sensor_trial_dir(model_name, loss_choice, dataset_name, sensor_pair, sensor_mode)
    MODELS_DIR = os.path.join(trial_dir, "models")
    LOGS_DIR = os.path.join(trial_dir, "logs")
    FIGURES_DIR = os.path.join(trial_dir, "figures")
    for d in [MODELS_DIR, LOGS_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)
        
    print(f"Loading Sensor Pair Dataset for {dataset_name} using sensor mode '{sensor_mode}' and sensor pair {sensor_pair}")
    base_dir_dataset = os.path.join(base_dir, dataset_name)
    train_index = os.path.join(base_dir_dataset, "train.csv")
    val_index   = os.path.join(base_dir_dataset, "val.csv")
    test_index  = os.path.join(base_dir_dataset, "test.csv")
    
    # Use the sensor_mode provided from ds_sensor_modes.
    train_dataset = EMG_dataset(
        train_index, lag=lag, n_ahead=n_ahead,
        input_sensor=sensor_mode, target_sensor="emg", randomize_legs=False,
        sensor_pair=sensor_pair,
        base_dir=base_dir
    )
    test_dataset = EMG_dataset(
        test_index, lag=lag, n_ahead=n_ahead,
        input_sensor=sensor_mode, target_sensor="emg", randomize_legs=False,
        sensor_pair=sensor_pair,
        base_dir=base_dir
    )
    val_dataset = EMG_dataset(
        val_index, lag=lag, n_ahead=n_ahead,
        input_sensor=sensor_mode, target_sensor="emg", randomize_legs=False,
        sensor_pair=sensor_pair,
        base_dir=base_dir
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    distribution_fig = os.path.join(FIGURES_DIR, f"{dataset_name}_sensor_pair_{sensor_pair}_distribution.png")
    train_dataset.plot_distribution(distribution_fig)
        # For sensor pair training, the input size is always 1.

    if model_name == "mdn":
        # For MDN, pass in the number of mixtures along with the other parameters.
        num_mixtures = 5  # Adjust as needed
        model = model_class(input_size=1,
                            hidden_size=256,
                            num_layers=5,
                            num_mixtures=num_mixtures,
                            num_classes=output_size,  # This forces 3 output signals
                            n_ahead=n_ahead).to(device)
    elif model_name in {"timeseries_transformer", "temporal_transformer", "informer"}:
        model = model_class(input_size=1, num_classes=output_size, n_ahead=n_ahead).to(device)
    elif model_name == "nbeats":
        model = model_class(input_size=1, num_stacks=3, num_blocks_per_stack=3, num_layers=4, 
                        hidden_size=256, output_size=1).to(device)
    else:
        model = model_class(input_size=1, hidden_size=256, num_layers=5, num_classes=1, n_ahead=n_ahead).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=fast_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch+1)/5 if epoch < 5 else final_lr/fast_lr)

    epoch_log_file = os.path.join(LOGS_DIR, f"loss_summary_sensor_{sensor_pair}_{model_name}.txt")
    trainer = Trainer(model=model, loss=loss_choice, optimizer=optimizer, scheduler=scheduler,
                      testloader=test_loader, fig_dir=FIGURES_DIR, model_type=model_name, device=device,
                      use_variation_penalty=True, alpha=1.0, var_threshold=0.01)
    trainer.epoch_log_file = epoch_log_file

    checkpoint_dir = os.path.join(trial_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_pattern = os.path.join(checkpoint_dir, f"model_sensor_{sensor_pair}_{model_name}_epoch_*.pth")
    checkpoint_files = glob.glob(checkpoint_pattern)
    start_epoch = 0
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(re.findall(r'epoch_(\d+)\.pth', x)[0]))
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch} for sensor pair {sensor_pair} {model_name} / {loss_choice} on {dataset_name} ...")

    for epoch in range(start_epoch, default_epochs):
        trainer.Training_Loop(train_loader)
        trainer.Validation_Loop(val_loader)
        trainer.step_scheduler()

        print(f"\nEpoch {epoch+1}/{default_epochs} for sensor pair {sensor_pair} {model_name} ({loss_choice}) on {dataset_name}")
        print("  Train Loss:", trainer.Metrics["Training Loss"][-1])
        print("  Valid Loss:", trainer.Metrics["Validation Loss"][-1])

        checkpoint_path = os.path.join(checkpoint_dir, f"model_sensor_{sensor_pair}_{model_name}_epoch_{epoch+1}.pth")
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
        print(f"  Checkpoint saved to {checkpoint_path}")

        trainer.save_first_10_windows(train_loader, epoch+1)

    final_model_path = os.path.join(MODELS_DIR, f"model_sensor_{sensor_pair}_{model_name}.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    trainer.Test_Model(test_loader)
    val_plot_path = os.path.join(trial_dir, "validation_results.png")
    trainer.plot_validation_results(val_loader, val_plot_path)

    test_results = trainer.test_results
    sample_idx = 0
    test_batch = next(iter(test_loader))
    X_sample, Y_sample, *_ = test_batch
    X_sample = X_sample[sample_idx].detach().cpu().numpy()
    Y_sample = Y_sample[sample_idx].detach().cpu().numpy()
    Y_pred = test_results["preds"][sample_idx].detach().cpu().numpy()
    channel = 0
    combined_true = np.concatenate([X_sample[:, channel], Y_sample[:, channel]])
    combined_pred = np.concatenate([X_sample[:, channel], Y_pred[:, channel]])

    plt.figure(figsize=(10, 6))
    plt.plot(combined_true, label="Ground Truth", marker="o")
    plt.plot(combined_pred, label="Prediction", marker="x", linestyle="--")
    plt.axvline(x=len(X_sample)-0.5, color="black", linestyle="--", label="Forecast Boundary")
    plt.xlabel("Time Step")
    plt.ylabel("Signal Value")
    plt.title(f"Combined Prediction vs Ground Truth (sensor pair {sensor_pair} {model_name} / {loss_choice}) on {dataset_name}")
    plt.legend()
    final_fig_path = os.path.join(trial_dir, "combined_prediction.png")
    plt.savefig(final_fig_path)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(trainer.Metrics["Training Loss"], label="Training Loss")
    plt.plot(trainer.Metrics["Validation Loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve (sensor pair {sensor_pair} {model_name} / {loss_choice}) on {dataset_name}")
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(trial_dir, "loss_curve.png")
    plt.savefig(loss_plot_path)
    plt.close()

    return trainer.Metrics

# ----------------------------------------------------------------------------------
# Main execution block
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Starting training on device: {device}")
    dataset_names = ["DS1", "DS2", "DS3", "DS4"]

        # Standard training runs.
    for loss_func in LOSS_TYPES:
        print(f"\n--- Starting standard training with loss function: {loss_func} ---")
        for ds in dataset_names:
            for model_name, model_cls in model_variants.items():
                print(f"\n=== Standard Training: {model_name} on {ds} with {loss_func} loss and input mode {input_mode} ===")
                run_training(model_cls, model_name, loss_func, ds)
        print(f"\n--- Starting sensor pair training with loss function: {loss_func} ---")
        for ds in dataset_names:
            # Loop over each sensor mode configuration for the dataset.
            for sensor_mode, available_channels in ds_sensor_modes[ds]:
                # Loop over valid sensor pair indices.
                for sensor in range(available_channels):
                    for model_name, model_cls in model_variants.items():
                        print(f"\n=== Sensor Pair Training: sensor pair {sensor} for {model_name} on {ds} with {loss_func} loss, sensor mode '{sensor_mode}' ===")
                        run_training_sensor_pair(model_cls, model_name, loss_func, ds, sensor_pair=sensor, sensor_mode=sensor_mode)
    print("All single sensor runs completed!")
    print("All training runs completed!")
