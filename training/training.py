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
from utils.datasets import EMG_dataset
from utils.Trainer import Trainer  

# Import model variants.
# Ensure that the new models (PatchTST, CrossFormer, DLinear) are also imported.
from models.models import (
    LSTMModel, RNNModel, GRUModel, TCNModel, TemporalTransformer,
    TimeSeriesTransformer, Informer, NBeats, DBN,
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
# ----------------------------------------------------------------------------------
def run_training(model_class, model_name, loss_choice, sensor_mode, n_ahead_val=None, target_sensor="emg"):
    print(f"Starting training: {model_name} (n_ahead = {n_ahead_val}) using input: {sensor_mode} with {loss_choice} loss")
    # Get the number of channels based on input sensor type.
    selected_channels = input_sizes[sensor_mode]

    # Our dataset is assumed to use CSV indices stored in 'train_index.csv', etc.
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

    # Determine if we should keep time data (only for Informer model)
    keep_time = model_name.lower() == "informer"

    # Instantiate the dataset.
    train_dataset = EMG_dataset(
        processed_index_csv=train_index, lag=lag, n_ahead=n_ahead_val,
        input_sensor=sensor_mode, target_sensor=target_sensor,
        base_dir=base_dir, keep_time=keep_time
    )
    val_dataset = EMG_dataset(
        processed_index_csv=val_index, lag=lag, n_ahead=n_ahead_val,
        input_sensor=sensor_mode, target_sensor=target_sensor,
        base_dir=base_dir, keep_time=keep_time
    )
    test_dataset = EMG_dataset(
        processed_index_csv=test_index, lag=lag, n_ahead=n_ahead_val,
        input_sensor=sensor_mode, target_sensor=target_sensor, 
        base_dir=base_dir, keep_time=keep_time
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Instantiate the model based on the model name.
    if model_name == "tcn":
        model = model_class(input_channels=selected_channels, num_classes=output_size, n_ahead=n_ahead_val).to(device)
    elif model_name == "timeseries_transformer":
        # TimeSeriesTransformer expects output_size
        model = model_class(input_size=selected_channels, output_size=output_size, n_ahead=n_ahead_val).to(device)
    elif model_name == "temporal_transformer":
        # TemporalTransformer expects num_classes
        model = model_class(input_size=selected_channels, num_classes=output_size, n_ahead=n_ahead_val).to(device)
    elif model_name == "nbeats":
        flattened_input_size = lag * selected_channels
        model = model_class(
            input_size=flattened_input_size, 
            num_stacks=3, 
            num_blocks_per_stack=3, 
            num_layers=4,
            hidden_size=256, 
            output_size=output_size, 
            n_ahead=n_ahead_val
        ).to(device)
    elif model_name == "informer":
        label_len = lag // 2  
        model = model_class(
            enc_in=selected_channels,            # encoder input dimension
            dec_in=selected_channels,            # decoder input dimension
            c_out=output_size,                   # output feature dimension
            seq_len=lag,                         # total input sequence length
            label_len=label_len,                 # decoder conditioning length
            out_len=n_ahead_val,                 # forecast horizon
            activation="relu",                   
            output_attention=False               
        ).to(device)
    elif model_name == "dbn":
        flattened_input_size = lag * selected_channels
        sizes = [flattened_input_size, 128, 128]
        model = model_class(sizes=sizes, output_dim=output_size, n_ahead=n_ahead_val).to(device)
        print("Starting DBN pretraining...")
        model.pretrain(train_loader, num_epochs=10, batch_size=batch_size, verbose=True)
    # --- New Models ---
    elif model_name == "patchtst":
        model = model_class(
            input_channels=selected_channels,
            patch_size=16,
            d_model=512,
            nhead=8,
            num_layers=3,
            dim_feedforward=2048,
            dropout=0.1,
            forecast_horizon=n_ahead_val,
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
            forecast_horizon=n_ahead_val,
            output_size=output_size
        ).to(device)
    elif model_name == "dlinear":
        model = model_class(
            seq_len=lag,
            forecast_horizon=n_ahead_val,
            num_channels=selected_channels,
            individual=False,
            moving_avg_kernel=25
        ).to(device)
    else:
        # Default to recurrent models (LSTM, RNN, GRU, etc.)
        model = model_class(input_size=selected_channels, hidden_size=256, num_layers=5,
                            num_classes=output_size, n_ahead=n_ahead_val).to(device)


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
    start_epoch = 1  # Default starting epoch is 1
    if epoch_checkpoint_files:
        latest_checkpoint_file = max(epoch_checkpoint_files, key=lambda x: int(re.findall(r'epoch_(\d+)\.pt', x)[0]))
        checkpoint = torch.load(latest_checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch} for {model_name}/{loss_choice} with input {sensor_mode} ...")
    else:
        print("No epoch checkpoints found. Starting training from epoch 1.")

    remaining_epochs = default_epochs - start_epoch + 1
    loss_curve = f'{trial_dir}_{n_ahead_val}_{loss_choice}.png'

    # Pass the start_epoch parameter to the Trainer's fit method.
    metrics = trainer.fit(train_loader=train_loader, 
                          val_loader=val_loader, 
                          epochs=default_epochs,
                          checkpoint_dir=MODELS_DIR, 
                          patience=10, 
                          min_delta=0.0,
                          loss_curve_path=loss_curve,
                          start_epoch=start_epoch)
    
    test_save_path = os.path.join(trial_dir, "test_results.png")
    trainer.Test_Model(test_loader, test_save_path)
    val_plot_path = os.path.join(trial_dir, "validation_results.png")
    trainer.plot_validation_results(val_loader, val_plot_path)

    final_model_path = os.path.join(MODELS_DIR, f"model_{model_name}.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # ------------------------------
    # ONNX Exporting (optional)
    # ------------------------------
    # model_for_export = model.module if hasattr(model, "module") else model
    # model_for_export.eval()
    # dummy_input = torch.randn(1, lag, selected_channels, device=device)
    # onnx_model_path = os.path.join(MODELS_DIR, f"model_{model_name}.onnx")
    # torch.onnx.export(
    #     model_for_export,
    #     dummy_input,
    #     onnx_model_path,
    #     input_names=['input'],
    #     output_names=['output'],
    #     dynamic_axes={
    #         'input': {0: 'batch_size', 1: 'seq_len'},
    #         'output': {0: 'batch_size', 1: 'seq_len'}
    #     },
    #     opset_version=14
    # )
    # print(f"ONNX model exported to {onnx_model_path}")
    
    return metrics

def train_model(model_name, model_cls, loss_func, sensor_mode, n_val, target_sensor):
    """Wrapper function to run training with proper logging."""
    print(f"\nStarting training for: {model_name} (sensor: {sensor_mode}, n_ahead: {n_val}, loss: {loss_func})")
    return run_training(model_cls, model_name, loss_func, sensor_mode, n_val, target_sensor)

# ----------------------------------------------------------------------------------
# DEVICE / PATH CONFIGURATION & HYPERPARAMETERS
# ----------------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the base folder for trials.
BASE_TRIAL_DIR = "/data1/dnicho26/Thesis/AI-Assisted-Gait-Restoration-for-Disabled-Individuals/trials"
os.makedirs(BASE_TRIAL_DIR, exist_ok=True)

# Define the base directory for your dataset.
base_dir = "/data1/dnicho26/EMG_DATASET/final-data/"

lag = 30         # Length of the sliding window.
n_ahead = 10     # Default forecast horizon.
batch_size = 12
default_epochs = 300
fast_lr = 1e-4
final_lr = 7e-4
output_size = 3
target_sensor = "emg"  

# Define input sizes (number of channels) for each sensor mode.
input_sizes = {"all": 21, "emg": 3, "acc": 9, "gyro": 9}

# Define loss function types.
LOSS_TYPES = ["huber", "mse", "smoothl1", "custom"]

# ----------------------------------------------------------------------------------
# Define model variants.
# The dictionary now includes the new models: patchtst, crossformer, and dlinear.
# ----------------------------------------------------------------------------------
model_variants = {
    "informer": Informer,
    "temporal_transformer": TemporalTransformer,
    "dbn": DBN,
    "nbeats": NBeats,
    "lstm": LSTMModel,
    "rnn": RNNModel,
    "gru": GRUModel,
    "tcn": TCNModel,
    "patchtst": PatchTST,
    "crossformer": CrossFormer,
    "dlinear": DLinear,
    "timeseries_transformer": TimeSeriesTransformer,
}

# ----------------------------------------------------------------------------------
# Main Execution Block: Loop over sensor modes, forecast horizons, loss functions, and models.
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Starting training on device: {device}")
    sensor_modes = ["emg", "all"]
    n_ahead_values = [5, 10, 15, 20]

    # Generate all experiment combinations.
    experiments = []
    for sensor_mode in sensor_modes:
        for n_val in n_ahead_values:
            for loss_func in LOSS_TYPES:
                for model_name, model_cls in model_variants.items():
                    experiments.append((model_name, model_cls, loss_func, sensor_mode, n_val, target_sensor))

    print(f"Total experiments: {len(experiments)}")
    
    # Run experiments sequentially.
    for model_name, model_cls, loss_func, sensor_mode, n_val, target_sensor in experiments:
        train_model(model_name, model_cls, loss_func, sensor_mode, n_val, target_sensor)
        print(f"Completed training for {model_name} with {loss_func} loss")

    print("\nAll training runs completed!")
