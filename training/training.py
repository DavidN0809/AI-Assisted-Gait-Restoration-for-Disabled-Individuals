import os
import sys
import glob
import re
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Append parent directory to locate modules (assumes a project folder structure)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import dataset, trainer, and all models.
from utils.datasets import EMG_dataset
from utils.Trainer import Trainer  
from models.models import (
    LSTMModel, RNNModel, GRUModel, TCNModel, HybridTransformerLSTM,
    LSTMFullSequence, LSTMAutoregressive, LSTMDecoder,
    TimeSeriesTransformer, TemporalTransformer, Informer, NBeats, MDN
)

import torch.optim as optim

# ----------------------------------------------------------------------------------
# DEVICE / PATH CONFIG
# ----------------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

BASE_TRIAL_DIR = "/data1/dnicho26/Thesis/AI-Assisted-Gait-Restoration-for-Disabled-Individuals/trials"
os.makedirs(BASE_TRIAL_DIR, exist_ok=True)

# ----------------------------------------------------------------------------------
# DATASET CONFIG
# ----------------------------------------------------------------------------------
base_dir_dataset = "/data1/dnicho26/EMG_DATASET/data/final-data"
train_index = os.path.join(base_dir_dataset, "train_index.csv")
val_index   = os.path.join(base_dir_dataset, "val_index.csv")
test_index  = os.path.join(base_dir_dataset, "test_index.csv")

# ----------------------------------------------------------------------------------
# HYPERPARAMETERS
# ----------------------------------------------------------------------------------
lag = 30
n_ahead = 10
batch_size = 12

default_epochs = 50
fast_lr = 1e-3     # initial "fast" LR
final_lr = 1e-6    # LR after 50 epochs

output_size = 3

# Only train on EMG signals (EMG-to-EMG)
input_mode = "emg"
input_sizes = {"emg": 3}

# List of custom loss functions to try
LOSS_TYPES = ["huber", "mse", "smoothl1"]

# Define all model variants as a dictionary mapping names to classes.
# Order: transformer-based models first, then new LSTM variants, then the remaining models.
model_variants = {
    "timeseries_transformer": TimeSeriesTransformer,
    "temporal_transformer": TemporalTransformer,
    "informer": Informer,
    "nbeats": NBeats,
    "lstmauto": LSTMAutoregressive,
    "lstmdecoder": LSTMDecoder,
    "lstmfull": LSTMFullSequence,
    "lstm": LSTMModel,
    "rnn": RNNModel,
    "gru": GRUModel,
    "tcn": TCNModel,
    "hyrbid": HybridTransformerLSTM,
    "mdn": MDN
}

# ----------------------------------------------------------------------------------
# UTILITY: CREATE TRIAL DIRECTORY
# ----------------------------------------------------------------------------------
def get_trial_dir(model_name, loss_type):
    trial_dir = os.path.join(BASE_TRIAL_DIR, model_name, loss_type)
    os.makedirs(trial_dir, exist_ok=True)
    return trial_dir

# ----------------------------------------------------------------------------------
# MAIN TRAINING FUNCTION
# ----------------------------------------------------------------------------------
def run_training(model_class, model_name, loss_choice):
    """
    - Creates a directory for the given model/loss combination.
    - Instantiates the datasets and dataloaders.
    - Builds the model, optimizer, and scheduler.
    - Trains for default_epochs epochs.
    - Saves checkpoints, plots, and final metrics.
    """
    trial_dir = get_trial_dir(model_name, loss_choice)

    # Create subdirectories for models, logs, and figures.
    MODELS_DIR = os.path.join(trial_dir, "models")
    LOGS_DIR = os.path.join(trial_dir, "logs")
    FIGURES_DIR = os.path.join(trial_dir, "figures")
    for d in [MODELS_DIR, LOGS_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)
        
    # ----------------------------------------------------------------------------------
    # DATASETS & DATALOADERS
    # ----------------------------------------------------------------------------------
    print("Loading Datasets")
    train_dataset = EMG_dataset(
        train_index, lag=lag, n_ahead=n_ahead,
        input_sensor=input_mode, target_sensor="emg", randomize_legs=True
    )
    val_dataset = EMG_dataset(
        val_index, lag=lag, n_ahead=n_ahead,
        input_sensor=input_mode, target_sensor="emg", randomize_legs=True
    )
    test_dataset = EMG_dataset(
        test_index, lag=lag, n_ahead=n_ahead,
        input_sensor=input_mode, target_sensor="emg", randomize_legs=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print("Train dataset length:", len(train_dataset))
    train_dataset.plot_distribution(FIGURES_DIR)

    # ----------------------------------------------------------------------------------
    # MODEL, OPTIMIZER, SCHEDULER
    # ----------------------------------------------------------------------------------
    # Use different constructor arguments based on the model type.
    if model_name in {"timeseries_transformer", "temporal_transformer", "informer"}:
        model = model_class(
            input_size=input_sizes[input_mode],
            num_classes=output_size,
            n_ahead=n_ahead
        ).to(device)

    elif model_name == "mdn":
        model = model_class(
            input_size=input_sizes[input_mode],
            hidden_size=256,
            num_layers=5,
            num_mixtures=3,       
            num_classes=output_size,
            n_ahead=n_ahead
        ).to(device)
    elif model_name == "nbeats":
        model = model_class(
            input_size=input_sizes[input_mode],
            num_stacks=3,
            num_blocks_per_stack=3,
            num_layers=4,
            hidden_size=256,
            output_size=output_size
        ).to(device)
    else:
        model = model_class(
            input_size=input_sizes[input_mode],
            hidden_size=256,
            num_layers=5,
            num_classes=output_size,
            n_ahead=n_ahead
        ).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # optimizer = torch.optim.Adam(model.parameters(), lr=fast_lr, weight_decay=1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=fast_lr, weight_decay=1e-5)

    # Use RAdam
    # optimizer = optim.RAdam(model.parameters(), lr=fast_lr, weight_decay=1e-5)

    def lr_schedule(epoch):
        warmup_epochs = 5  # Adjust as needed
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # linearly increase lr
        else:
            # After warmup, decay as originally planned.
            return final_lr / fast_lr

    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
  

    # ----------------------------------------------------------------------------------
    # MODEL TRAINER
    # ----------------------------------------------------------------------------------
    epoch_log_file = os.path.join(LOGS_DIR, f"loss_summary_{input_mode}_{model_name}.txt")
    trainer = Trainer(
        model=model,
        loss=loss_choice,
        optimizer=optimizer,
        scheduler=scheduler,
        testloader=test_loader,
        fig_dir=FIGURES_DIR,
        model_type=model_name,
        device=device,
        use_variation_penalty=True,
        alpha=1.0,
        var_threshold=0.01
    )
    trainer.epoch_log_file = epoch_log_file

    # ----------------------------------------------------------------------------------
    # CHECK FOR EXISTING CHECKPOINTS (OPTIONAL)
    # ----------------------------------------------------------------------------------
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
        print(f"Resuming from epoch {start_epoch} for {model_name} / {loss_choice} ...")

    # ----------------------------------------------------------------------------------
    # TRAINING LOOP
    # ----------------------------------------------------------------------------------
    for epoch in range(start_epoch, default_epochs):
        trainer.Training_Loop(train_loader)
        trainer.Validation_Loop(val_loader)
        trainer.step_scheduler()

        print(f"\nEpoch {epoch+1}/{default_epochs} for {model_name} ({loss_choice})")
        print("  Train Loss:", trainer.Metrics["Training Loss"][-1])
        print("  Valid Loss:", trainer.Metrics["Validation Loss"][-1])

        checkpoint_path = os.path.join(checkpoint_dir, f"model_{input_mode}_{model_name}_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"  Checkpoint saved to {checkpoint_path}")

        # trainer.plot_predictions(epoch)
        trainer.save_first_10_windows(train_loader, epoch+1)

    # ----------------------------------------------------------------------------------
    # SAVE FINAL MODEL
    # ----------------------------------------------------------------------------------
    final_model_path = os.path.join(MODELS_DIR, f"model_{input_mode}_{model_name}.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    # ----------------------------------------------------------------------------------
    # EVALUATE ON TEST SET
    # ----------------------------------------------------------------------------------
    trainer.Test_Model(test_loader)

    # ----------------------------------------------------------------------------------
    # PLOT FULL VALIDATION RESULTS
    # ----------------------------------------------------------------------------------
    val_plot_path = os.path.join(trial_dir, "validation_results.png")
    trainer.plot_validation_results(val_loader, val_plot_path)

    # ----------------------------------------------------------------------------------
    # PLOT A SINGLE COMBINED PREDICTION (SAMPLE = 0, CHANNEL = 0)
    # ----------------------------------------------------------------------------------
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
    plt.title(f"Combined Prediction vs Ground Truth ({model_name} / {loss_choice})")
    plt.legend()
    final_fig_path = os.path.join(trial_dir, "combined_prediction.png")
    plt.savefig(final_fig_path)
    plt.close()

    # ----------------------------------------------------------------------------------
    # PLOT TRAIN/VALID LOSS CURVES
    # ----------------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(trainer.Metrics["Training Loss"], label="Training Loss")
    plt.plot(trainer.Metrics["Validation Loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve ({model_name} / {loss_choice})")
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(trial_dir, "loss_curve.png")
    plt.savefig(loss_plot_path)
    plt.close()

    return trainer.Metrics

# ----------------------------------------------------------------------------------
# MAIN SCRIPT: TRAIN EACH MODEL WITH EACH LOSS TYPE
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Starting training on device: {device}")
    for loss_func in LOSS_TYPES:
        # Iterate over the models in the order defined in the dictionary.
        for model_name, model_cls in model_variants.items():
            print(f"\n=== Training {model_name} with {loss_func} loss and input mode {input_mode} ===")
            run_training(model_cls, model_name, loss_func)
    print("All training runs completed!")
