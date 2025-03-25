"""
tune_models.py

This script performs hyperparameter tuning for EMG time-series forecasting models.
It supports both transformer-based models and LSTM-based models.
Multiple trials run concurrently (using Optuna’s n_jobs) and models are wrapped in DataParallel
to use multiple GPUs.
Best trial results for each loss type are saved to a text file ("tuning_results.txt").
The script also generates a full validation plot via the ModelTrainer's plot_validation_results.
"""

import os
import sys
import glob
import re
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import optuna
import numpy as np

# Append parent directory to locate modules (assumes a project folder structure)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.datasets import EMG_dataset
from utils.Trainer import ModelTrainer, CustomEMGLoss
from models.models import (
    BasicLSTM, BidirectionalLSTM, ResidualLSTM, AttentionLSTM,
    TimeSeriesTransformer, TemporalFusionTransformer, Informer, NBeats
)

# Set device; DataParallel will use all available GPUs.
device = "cuda" if torch.cuda.is_available() else "cpu"

###############################################################################
# CONFIGURATION (hardcoded settings)
###############################################################################
BASE_TRIAL_DIR = "/data1/dnicho26/Thesis/AI-Assisted-Gait-Restoration-for-Disabled-Individuals/trials"
os.makedirs(BASE_TRIAL_DIR, exist_ok=True)

# Dataset paths
base_dir_dataset = "/data1/dnicho26/EMG_DATASET/data/final-proc-server/index_files"
train_index = os.path.join(base_dir_dataset, "index_train_treadmill.csv")
val_index   = os.path.join(base_dir_dataset, "index_val_treadmill.csv")
test_index  = os.path.join(base_dir_dataset, "index_test_treadmill.csv")

# Common hyperparameters for tuning runs
lag = 60
n_ahead = 5
batch_size = 128
# Use a lower number of epochs during tuning; final training script will run 150+ epochs.
default_epochs = 50
default_lr = 0.0007
output_size = 3

input_configs = ["all", "emg", "acc", "gyro"]
input_sizes   = {"all": 21, "emg": 3, "acc": 9, "gyro": 9}
input_mode = "all"  # choose the sensor configuration

# List of loss types to try
LOSS_TYPES = ["logcosh", "huber", "SmoothL1Loss"]
loss_choice = None  # This global variable will be updated in the outer loop

# Choose model family: either "lstm" or "transformer"
model_family = "lstm"  # Change to "transformer" to tune transformer-based models or lstm

# Optuna tuning settings
n_trials = 10
n_jobs = 1

###############################################################################
# Model Variant Dictionaries
###############################################################################
lstm_variants = {
    "BasicLSTM": BasicLSTM,
    "BidirectionalLSTM": BidirectionalLSTM,
    "ResidualLSTM": ResidualLSTM,
    "AttentionLSTM": AttentionLSTM
}

transformer_variants = {
    "TimeSeriesTransformer": TimeSeriesTransformer,
    "TemporalFusionTransformer": TemporalFusionTransformer,
    "Informer": Informer,
    "NBeats": NBeats
}

###############################################################################
# Utility: Create Trial Directory
###############################################################################
def get_trial_dir(trial_identifier, model_name, loss_type):
    trial_dir = os.path.join(BASE_TRIAL_DIR, model_name, loss_type, f"trial_{trial_identifier}")
    os.makedirs(trial_dir, exist_ok=True)
    return trial_dir

###############################################################################
# Run Training for LSTM-based Models
###############################################################################
def run_training_lstm(model_class, hyperparams, trial_dir):
    # Create subdirectories for saving models, logs, and figures.
    MODELS_DIR = os.path.join(trial_dir, "models")
    LOGS_DIR = os.path.join(trial_dir, "logs")
    FIGURES_DIR = os.path.join(trial_dir, "figures")
    for d in [MODELS_DIR, LOGS_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)
    
    # Prepare datasets and data loaders.
    train_dataset = EMG_dataset(train_index, lag=lag, n_ahead=n_ahead,
                                input_sensor=input_mode, target_sensor="emg", randomize_legs=True)
    val_dataset = EMG_dataset(val_index, lag=lag, n_ahead=n_ahead,
                              input_sensor=input_mode, target_sensor="emg", randomize_legs=True)
    test_dataset = EMG_dataset(test_index, lag=lag, n_ahead=n_ahead,
                               input_sensor=input_mode, target_sensor="emg", randomize_legs=True)
    trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    validLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    testLoader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    input_size_model = input_sizes[input_mode]
    model = model_class(input_size_model, hyperparams["hidden_size"], hyperparams["num_layers"],
                        output_size, n_ahead).to(device)
    
    # If dropout tuning is desired for LSTM models and the model has a dropout attribute, update it.
    if "dropout" in hyperparams:
        if hasattr(model, "dropout"):
            if isinstance(model, torch.nn.DataParallel):
                model.module.dropout = nn.Dropout(hyperparams["dropout"])
            else:
                model.dropout = nn.Dropout(hyperparams["dropout"])
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"], weight_decay=1e-5)
    loss = CustomEMGLoss(sensor_indices=[0,1,2],
                         quantile=0.9,
                         input_loss_type=loss_choice,
                         input_loss_weight=0.5,
                         derivative_weight=0.1,
                         correlation_weight=0.1,
                         forecast_start_weight=0.1)
    
    # Save loss config and set up log file.
    loss_config_path = os.path.join(trial_dir, "loss_config.txt")
    with open(loss_config_path, "w") as f:
        f.write(f"Loss type: {loss_choice}\nLoss parameters: quantile=0.9, input_loss_weight=0.5, "
                f"derivative_weight=0.1, correlation_weight=0.1, forecast_start_weight=0.1\n")
    epoch_log_file = os.path.join(LOGS_DIR, f"loss_summary_{input_mode}_{hyperparams['model_name']}.txt")
    
    trainer = ModelTrainer(
        model=model,
        loss=loss,
        optimizer=optimizer,
        accuracy=None,
        model_type="Regression",
        model_name=hyperparams["model_name"],
        input_mode=input_mode,
        device=device,
        noPrint=False,
        flatten_output=False,
        testloader=testLoader,
        fig_dir=FIGURES_DIR
    )
    trainer.epoch_log_file = epoch_log_file
    
    # Check for existing checkpoints.
    checkpoint_dir = os.path.join(trial_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_pattern = os.path.join(checkpoint_dir, f"model_{input_mode}_{hyperparams['model_name']}_epoch_*.pth")
    checkpoint_files = glob.glob(checkpoint_pattern)
    start_epoch = 0
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(re.findall(r'epoch_(\d+)\.pth', x)[0]))
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    t0 = datetime.now()
    trainer.fit(trainLoader, validLoader, epochs=default_epochs, start_epoch=start_epoch, checkpoint_dir=checkpoint_dir)
    t1 = datetime.now()
    
    final_model_path = os.path.join(MODELS_DIR, f"model_{input_mode}_{hyperparams['model_name']}.pt")
    torch.save(model.state_dict(), final_model_path)
    trainer.Test_Model(testLoader)
    
    # Generate a full validation plot.
    val_plot_path = os.path.join(trial_dir, "validation_results.png")
    trainer.plot_validation_results(validLoader, val_plot_path)
    
    # Also save a combined prediction plot.
    test_results = trainer.test_results
    sample_idx = 0
    test_batch = next(iter(testLoader))
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
    plt.title(f"Combined Prediction vs Ground Truth for {hyperparams['model_name']}")
    plt.legend()
    final_fig_path = os.path.join(trial_dir, "combined_prediction.png")
    plt.savefig(final_fig_path)
    plt.close()
    
    return trainer.Metrics

###############################################################################
# Run Training for Transformer-based Models
###############################################################################
def run_training_transformer(model_class, hyperparams, trial_dir):
    # Create subdirectories for saving models, logs, and figures.
    MODELS_DIR = os.path.join(trial_dir, "models")
    LOGS_DIR = os.path.join(trial_dir, "logs")
    FIGURES_DIR = os.path.join(trial_dir, "figures")
    for d in [MODELS_DIR, LOGS_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)
    
    # Prepare datasets and data loaders.
    train_dataset = EMG_dataset(train_index, lag=lag, n_ahead=n_ahead,
                                input_sensor=input_mode, target_sensor="emg", randomize_legs=True)
    val_dataset = EMG_dataset(val_index, lag=lag, n_ahead=n_ahead,
                              input_sensor=input_mode, target_sensor="emg", randomize_legs=True)
    test_dataset = EMG_dataset(test_index, lag=lag, n_ahead=n_ahead,
                               input_sensor=input_mode, target_sensor="emg", randomize_legs=True)
    trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    validLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    testLoader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    input_size_model = input_sizes[input_mode]
    # Instantiate the transformer model based on its type.
    if hyperparams["model_name"] == "TimeSeriesTransformer":
        model = model_class(input_size_model,
                            hyperparams["num_heads"],
                            hyperparams["transformer_layers"],
                            hyperparams["transformer_hidden_dim"],
                            output_size,
                            dropout=hyperparams["dropout"],
                            n_ahead=n_ahead).to(device)
    elif hyperparams["model_name"] == "TemporalFusionTransformer":
        model = model_class(input_size_model,
                            hyperparams["hidden_dim"],
                            hyperparams["num_heads"],
                            hyperparams["transformer_layers"],
                            output_size,
                            dropout=hyperparams["dropout"],
                            n_ahead=n_ahead).to(device)
    elif hyperparams["model_name"] == "Informer":
        model = model_class(input_size_model,
                            hyperparams["d_model"],
                            hyperparams["num_heads"],
                            hyperparams["encoder_layers"],
                            output_size,
                            n_ahead=n_ahead,
                            dropout=hyperparams["dropout"]).to(device)
    else:  # NBeats
        model = model_class(input_size_model,
                            hyperparams["hidden_dim"],
                            output_size,
                            n_ahead,
                            num_blocks=hyperparams["num_blocks"]).to(device)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"], weight_decay=1e-5)
    loss = CustomEMGLoss(sensor_indices=[0,1,2],
                         quantile=0.9,
                         input_loss_type=loss_choice,
                         input_loss_weight=0.5,
                         derivative_weight=0.1,
                         correlation_weight=0.1,
                         forecast_start_weight=0.1)
    
    loss_config_path = os.path.join(trial_dir, "loss_config.txt")
    with open(loss_config_path, "w") as f:
        f.write(f"Loss type: {loss_choice}\nLoss parameters: quantile=0.9, input_loss_weight=0.5, "
                f"derivative_weight=0.1, correlation_weight=0.1, forecast_start_weight=0.1\n")
    epoch_log_file = os.path.join(LOGS_DIR, f"loss_summary_{input_mode}_{hyperparams['model_name']}.txt")
    
    trainer = ModelTrainer(
        model=model,
        loss=loss,
        optimizer=optimizer,
        accuracy=None,
        model_type="Regression",
        model_name=hyperparams["model_name"],
        input_mode=input_mode,
        device=device,
        noPrint=False,
        flatten_output=False,
        testloader=testLoader,
        fig_dir=FIGURES_DIR
    )
    trainer.epoch_log_file = epoch_log_file
    
    # Check for existing checkpoints.
    checkpoint_dir = os.path.join(trial_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_pattern = os.path.join(checkpoint_dir, f"model_{input_mode}_{hyperparams['model_name']}_epoch_*.pth")
    checkpoint_files = glob.glob(checkpoint_pattern)
    start_epoch = 0
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(re.findall(r'epoch_(\d+)\.pth', x)[0]))
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    t0 = datetime.now()
    trainer.fit(trainLoader, validLoader, epochs=default_epochs, start_epoch=start_epoch, checkpoint_dir=checkpoint_dir)
    t1 = datetime.now()
    
    final_model_path = os.path.join(MODELS_DIR, f"model_{input_mode}_{hyperparams['model_name']}.pt")
    torch.save(model.state_dict(), final_model_path)
    trainer.Test_Model(testLoader)
    
    # Generate full validation plot.
    val_plot_path = os.path.join(trial_dir, "validation_results.png")
    trainer.plot_validation_results(validLoader, val_plot_path)
    
    # Generate combined prediction plot.
    test_results = trainer.test_results
    sample_idx = 0
    test_batch = next(iter(testLoader))
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
    plt.title(f"Combined Prediction vs Ground Truth for {hyperparams['model_name']}")
    plt.legend()
    final_fig_path = os.path.join(trial_dir, "combined_prediction.png")
    plt.savefig(final_fig_path)
    plt.close()
    
    return trainer.Metrics

###############################################################################
# Objective Functions for Optuna
###############################################################################
def lstm_objective(trial):
    # Tune over LSTM variant and hyperparameters.
    model_type = trial.suggest_categorical("model_type", list(lstm_variants.keys()))
    hidden_size = trial.suggest_int("hidden_size", 64, 256, step=32)
    num_layers = trial.suggest_int("num_layers", 1, 4)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    # Optionally tune dropout if model supports it.
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    
    hyperparams = {
        "model_name": model_type,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "lr": lr,
        "dropout": dropout
    }
    trial_dir = get_trial_dir(trial.number, hyperparams["model_name"], loss_choice)
    metrics = run_training_lstm(lstm_variants[model_type], hyperparams, trial_dir)
    val_loss = metrics["Validation Loss"][-1]
    return val_loss

def transformer_objective(trial):
    # Tune over transformer variant and hyperparameters.
    model_type = trial.suggest_categorical("model_name", list(transformer_variants.keys()))
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.05, 0.3)
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
    
    if model_type == "TimeSeriesTransformer":
        transformer_layers = trial.suggest_int("transformer_layers", 1, 4)
        transformer_hidden_dim = trial.suggest_int("transformer_hidden_dim", 64, 256, step=32)
        hyperparams = {
            "model_name": model_type,
            "lr": lr,
            "dropout": dropout,
            "num_heads": num_heads,
            "transformer_layers": transformer_layers,
            "transformer_hidden_dim": transformer_hidden_dim
        }
    elif model_type == "TemporalFusionTransformer":
        transformer_layers = trial.suggest_int("transformer_layers", 1, 4)
        hidden_dim = trial.suggest_int("hidden_dim", 64, 256, step=32)
        hyperparams = {
            "model_name": model_type,
            "lr": lr,
            "dropout": dropout,
            "num_heads": num_heads,
            "transformer_layers": transformer_layers,
            "hidden_dim": hidden_dim
        }
    elif model_type == "Informer":
        encoder_layers = trial.suggest_int("encoder_layers", 1, 4)
        d_model = trial.suggest_int("d_model", 64, 256, step=32)
        hyperparams = {
            "model_name": model_type,
            "lr": lr,
            "dropout": dropout,
            "num_heads": num_heads,
            "encoder_layers": encoder_layers,
            "d_model": d_model
        }
    else:  # NBeats
        num_blocks = trial.suggest_int("num_blocks", 1, 4)
        hidden_dim = trial.suggest_int("hidden_dim", 64, 256, step=32)
        hyperparams = {
            "model_name": model_type,
            "lr": lr,
            "dropout": dropout,
            "hidden_dim": hidden_dim,
            "num_blocks": num_blocks
        }
    trial_dir = get_trial_dir(trial.number, hyperparams["model_name"], loss_choice)
    metrics = run_training_transformer(transformer_variants[hyperparams["model_name"]], hyperparams, trial_dir)
    val_loss = metrics["Validation Loss"][-1]
    return val_loss

###############################################################################
# MAIN TUNING LOOP (no CLI parsing)
###############################################################################
results_file = f"tuning_results_{model_family}.txt"
with open(results_file, "w") as rf:
    rf.write(f"Tuning Results for model family: {model_family}\n")
    rf.write(f"Date: {datetime.now()}\n\n")

for loss_func in LOSS_TYPES:
    loss_choice = loss_func  # update the global loss type
    if model_family == "lstm":
        study = optuna.create_study(direction="minimize")
        study.optimize(lstm_objective, n_trials=n_trials, n_jobs=n_jobs)
    else:
        study = optuna.create_study(direction="minimize")
        study.optimize(transformer_objective, n_trials=n_trials, n_jobs=n_jobs)
    best_trial = study.best_trial
    with open(results_file, "a") as rf:
        rf.write(f"Loss Type: {loss_choice} ({model_family})\n")
        rf.write(f"Best Value (Validation Loss): {best_trial.value}\n")
        rf.write("Best Parameters:\n")
        for key, value in best_trial.params.items():
            rf.write(f"  {key}: {value}\n")
        rf.write("\n")
