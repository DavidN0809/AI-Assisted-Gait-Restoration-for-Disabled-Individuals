#!/usr/bin/env python3
"""
training-non-normalized.py - Training Script Using Non-Normalized Data

This script uses a non-normalized dataset (EMG_dataset_window_norm) and otherwise
follows a similar structure to training.py. A summary CSV is generated at the end.
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.datasets import EMG_dataset_window_norm
from utils.Trainer import Trainer  
from models.models import (
    LSTMModel, RNNModel, GRUModel, TCNModel,
    TimeSeriesTransformer, TemporalTransformer, Informer, NBeats, DBN,
    PatchTST, CrossFormer, DLinear, HybridLSTMTransformer
)
from utils.metrics import save_summary_csv

def get_trial_dir(model_name, loss_type, sensor_mode, n_ahead_val=None):
    if n_ahead_val is not None:
        trial_dir = os.path.join(BASE_TRIAL_DIR, sensor_mode, model_name, str(n_ahead_val), loss_type)
    else:
        trial_dir = os.path.join(BASE_TRIAL_DIR, sensor_mode, model_name, loss_type)
    os.makedirs(trial_dir, exist_ok=True)
    return trial_dir

def run_training(model_class, model_name, loss_choice, sensor_mode, n_ahead_val=None, target_sensor="emg"):
    print(f"Starting training: {model_name} (n_ahead = {n_ahead_val}) using input: {sensor_mode} with {loss_choice} loss")
    selected_channels = input_sizes[sensor_mode]

    train_index = os.path.join(base_dir, "train_index.csv")
    val_index   = os.path.join(base_dir, "val_index.csv")
    test_index  = os.path.join(base_dir, "test_index.csv")

    trial_dir = get_trial_dir(model_name, loss_choice, sensor_mode, n_ahead_val)
    MODELS_DIR = os.path.join(trial_dir, "models")
    LOGS_DIR = os.path.join(trial_dir, "logs")
    FIGURES_DIR = os.path.join(trial_dir, "figures")
    for d in [MODELS_DIR, LOGS_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)

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

    if model_name == "tcn":
        model = model_class(input_channels=selected_channels, num_classes=output_size, n_ahead=n_ahead_val).to(device)
    elif model_name == "timeseries_transformer":
        model = model_class(input_size=selected_channels, output_size=output_size, n_ahead=n_ahead_val).to(device)
    elif model_name == "temporal_transformer":
        model = model_class(input_size=selected_channels, num_classes=output_size, n_ahead=n_ahead_val).to(device)
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
    elif model_name == "hybridlstmtransformer":
        model = model_class(input_size=selected_channels, hidden_size=256, num_layers=5,
                            num_classes=output_size, n_ahead=n_ahead_val).to(device)
    elif model_name == "dbn":
        flattened_input_size = lag * selected_channels
        sizes = [flattened_input_size, 128, 64]
        model = model_class(sizes=sizes, output_dim=output_size, n_ahead=n_ahead_val).to(device)
    else:
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

    checkpoint_dir = os.path.join(trial_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    epoch_checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt")
    epoch_checkpoint_files = glob.glob(epoch_checkpoint_pattern)
    start_epoch = 1
    if epoch_checkpoint_files:
        latest_checkpoint_file = max(epoch_checkpoint_files, key=lambda x: int(re.findall(r'epoch_(\d+)\.pt', x)[0]))
        checkpoint = torch.load(latest_checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("No epoch checkpoints found. Starting training from epoch 1.")

    trainer.fit(train_loader, val_loader, epochs=default_epochs - start_epoch, patience=10, min_delta=0.0,
                checkpoint_dir=checkpoint_dir, loss_curve_path=os.path.join(trial_dir, f"{n_ahead_val}_{loss_choice}.png"))
    
    test_save_path = os.path.join(trial_dir, "test_results.png")
    trainer.Test_Model(test_loader, test_save_path)
    val_plot_path = os.path.join(trial_dir, "validation_results.png")
    trainer.plot_validation_results(val_loader, val_plot_path)

    final_model_path = os.path.join(MODELS_DIR, f"model_{model_name}.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # ONNX Exporting
    model_for_export = model.module if hasattr(model, "module") else model
    model_for_export.eval()
    dummy_input = torch.randn(1, lag, selected_channels, device=device)
    onnx_model_path = os.path.join(MODELS_DIR, f"model_{model_name}.onnx")
    torch.onnx.export(
        model_for_export,
        dummy_input,
        onnx_model_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size', 1: 'seq_len'},
                      'output': {0: 'batch_size', 1: 'seq_len'}},
        opset_version=14
    )
    print(f"ONNX model exported to {onnx_model_path}")
    
    return trainer.Metrics

def train_model(model_name, model_cls, loss_func, sensor_mode, n_val, target_sensor):
    try:
        print(f"\nStarting training for: {model_name} (sensor: {sensor_mode}, n_ahead: {n_val}, loss: {loss_func})")
        return run_training(model_cls, model_name, loss_func, sensor_mode, n_val, target_sensor)
    except Exception as e:
        print(f"Error training {model_name}: {str(e)}")
        return None

device = "cuda" if torch.cuda.is_available() else "cpu"
BASE_TRIAL_DIR = "/data1/dnicho26/Thesis/AI-Assisted-Gait-Restoration-for-Disabled-Individuals/trials-not-norm"
os.makedirs(BASE_TRIAL_DIR, exist_ok=True)
base_dir = "/data1/dnicho26/EMG_DATASET/final-data-not-norm/"
lag = 30
n_ahead = 10
batch_size = 12
default_epochs = 300
fast_lr = 1e-4
final_lr = 7e-4
output_size = 3
target_sensor = "emg"
input_sizes = {"all": 21, "emg": 3, "acc": 9, "gyro": 9}
LOSS_TYPES = ["huber", "custom"]

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
    "hybridlstmtransformer": HybridLSTMTransformer
}

if __name__ == "__main__":
    experiments = []
    for sensor_mode in ["emg"]:
        for n_val in [10, 15, 20]:
            for loss_func in LOSS_TYPES:
                for model_name, model_cls in model_variants.items():
                    experiments.append((model_name, model_cls, loss_func, sensor_mode, n_val, target_sensor))
    print(f"Total experiments: {len(experiments)}")
    
    results = []
    for model_name, model_cls, loss_func, sensor_mode, n_val, target_sensor in experiments:
        metrics_dict = train_model(model_name, model_cls, loss_func, sensor_mode, n_val, target_sensor)
        if metrics_dict is not None:
            metrics_dict.update({
                'model': model_name,
                'loss_type': loss_func,
                'sensor_mode': sensor_mode,
                'n_ahead': n_val
            })
            results.append(metrics_dict)
            print(f"Completed training for {model_name} with {loss_func} loss")
        else:
            print(f"Failed training for {model_name} with {loss_func} loss")
    save_summary_csv(results, BASE_TRIAL_DIR)
    print("\nAll training runs completed!")
