# lstm.py
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# For data normalization
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
from joblib import Parallel, delayed

# Import your dataset class (now modified to downsample raw data before windowing)
from utils.datasets import EMG_dataset
# Import your LSTM models and transformer (see updated models.py)
from models.models import BasicLSTM, BidirectionalLSTM, ResidualLSTM, AttentionLSTM, TimeSeriesTransformer
# Import your trainer
from utils.Trainer import ModelTrainer

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# General hyperparameters
train_index = r"D:\UNC Charlotte Dropbox\orgs-ecgr-QuantitativeImagingandAILaboratory\processed-test\index_train.csv"
val_index = r"D:\UNC Charlotte Dropbox\orgs-ecgr-QuantitativeImagingandAILaboratory\processed-test\index_val.csv"
test_index = r"D:\UNC Charlotte Dropbox\orgs-ecgr-QuantitativeImagingandAILaboratory\processed-test\index_test.csv"

base_dir = r"C:\Users\alway\OneDrive\Documents\GitHub\AI-Assisted-Gait-Restoration-for-Disabled-Individuals"

scale_y_flag = False

lag = 30
n_ahead = 10
batch_size = 512
epochs = 300
lr = 0.0007
hidden_size = 128
num_layers = 5
output_size = 3  # Always predict 3 channels (target leg EMG)

# (EMG_ORIG_RATE and EMG_TARGET_RATE are no longer used in this file)
EMG_ORIG_RATE = 1200
EMG_TARGET_RATE = 148

# Ensure checkpoint directory exists
checkpoint_dir = os.path.join(base_dir, "models", "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# Ensure trained models directory exists
trained_dir = os.path.join(base_dir, "models", "trained")
os.makedirs(trained_dir, exist_ok=True)

# Define input configurations and corresponding sizes
input_configs = ["all", "emg", "acc", "gyro"]
input_sizes = {"all": 21, "emg": 3, "acc": 9, "gyro": 9}

log_base_dir = os.path.join(base_dir, "logs")
os.makedirs(log_base_dir, exist_ok=True)

fig_base_dir = os.path.join(base_dir, "figures", "training")
os.makedirs(fig_base_dir, exist_ok=True)

results = {}

##############################################################################
# Combined approach using parallel processing and incremental statistics
##############################################################################
def compute_chunk_statistics(dataset, chunk_indices):
    n = 0
    mean = None
    M2 = None  # Sum of squared differences
    for i in chunk_indices:
        X, _, _ = dataset[i]
        # Flatten X: shape (lag * in_features)
        X_flat = X.view(-1, X.shape[-1]).numpy()  # shape: (n_samples, n_features)
        current_n = X_flat.shape[0]
        current_mean = np.mean(X_flat, axis=0)
        current_var = np.var(X_flat, axis=0)
        current_M2 = current_var * current_n  # M2 for current batch

        if mean is None:
            mean = current_mean
            M2 = current_M2
            n = current_n
        else:
            delta = current_mean - mean
            total_n = n + current_n
            mean = (n * mean + current_n * current_mean) / total_n
            M2 = M2 + current_M2 + (delta ** 2) * n * current_n / total_n
            n = total_n
    return n, mean, M2

def gather_all_inputs_combined(dataset, n_jobs=4):
    indices = np.array_split(np.arange(len(dataset)), n_jobs)
    results_chunk = Parallel(n_jobs=n_jobs)(
        delayed(compute_chunk_statistics)(dataset, chunk) for chunk in indices
    )
    total_n = 0
    total_mean = None
    total_M2 = None
    for n, mean, M2 in results_chunk:
        if total_mean is None:
            total_n = n
            total_mean = mean
            total_M2 = M2
        else:
            delta = mean - total_mean
            new_total_n = total_n + n
            total_mean = (total_n * total_mean + n * mean) / new_total_n
            total_M2 = total_M2 + M2 + (delta ** 2) * total_n * n / new_total_n
            total_n = new_total_n
    variance = total_M2 / total_n
    scaler = StandardScaler()
    scaler.mean_ = total_mean
    scaler.var_ = variance
    scaler.scale_ = np.sqrt(variance)
    scaler.n_features_in_ = total_mean.shape[0]
    return scaler

##############################################################################
# Wrapper dataset to apply a fitted scaler on-the-fly to each sample's input X
##############################################################################
class ScaledDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, scaler_x, scaler_y=None):
        self.base_dataset = base_dataset
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        X, Y, action = self.base_dataset[idx]
        shape = X.shape  # (lag, in_features) where lag may have changed due to downsampling
        X_2d = X.view(-1, shape[-1]).numpy()
        X_scaled_2d = self.scaler_x.transform(X_2d)
        X_scaled = torch.from_numpy(X_scaled_2d).view(shape)
        
        if self.scaler_y is not None:
            y_shape = Y.shape  # (n_ahead, out_channels)
            Y_2d = Y.view(-1, y_shape[-1]).numpy()
            Y_scaled_2d = self.scaler_y.transform(Y_2d)
            Y_scaled = torch.from_numpy(Y_scaled_2d).view(y_shape)
        else:
            Y_scaled = Y
        return X_scaled, Y_scaled, action

##############################################################################
# Utility: Gather all target windows Y from a dataset to fit the Y scaler
##############################################################################
def gather_all_targets(dataset):
    all_Y = []
    for i in range(len(dataset)):
        _, Y, _ = dataset[i]
        Y_2d = Y.view(-1, Y.shape[-1]).numpy()
        all_Y.append(Y_2d)
    all_Y = np.concatenate(all_Y, axis=0)
    return all_Y

# Main loop over input configurations
for input_mode in input_configs:
    print("\n========================================")
    print("Training model with input configuration:", input_mode)
    
    epoch_log_file = os.path.join(log_base_dir, f"loss_summary_{input_mode}.txt")
    with open(epoch_log_file, "w") as f:
        f.write(f"Loss summary for input config: {input_mode}\n\n")
    
    fig_dir = os.path.join(fig_base_dir, input_mode)
    os.makedirs(fig_dir, exist_ok=True)
    
    print("Loading Train Dataset \n")
    # Directly load the dataset. Downsampling is now done inside datasets.py.
    train_dataset = EMG_dataset(train_index, lag=lag, n_ahead=n_ahead,
                                input_sensor=input_mode, target_sensor="emg", randomize_legs=False)
    X_ex, Y_ex, action_ex = train_dataset[0]
    print("Example input and target shapes:", X_ex.shape, Y_ex.shape, "Action:", action_ex)
    print(f"Train dataset length: {len(train_dataset)}")

    print("Loading Val Dataset \n")
    val_dataset = EMG_dataset(val_index, lag=lag, n_ahead=n_ahead,
                              input_sensor=input_mode, target_sensor="emg", randomize_legs=False)
    X_ex, Y_ex, action_ex = val_dataset[0]
    print("Example input and target shapes:", X_ex.shape, Y_ex.shape, "Action:", action_ex)
    print(f"Val dataset length: {len(val_dataset)}")

    print("Loading Test Dataset \n")
    test_dataset = EMG_dataset(test_index, lag=lag, n_ahead=n_ahead,
                               input_sensor=input_mode, target_sensor="emg", randomize_legs=False)
    X_ex, Y_ex, action_ex = test_dataset[0]
    print("Example input and target shapes:", X_ex.shape, Y_ex.shape, "Action:", action_ex)
    print(f"Test dataset length: {len(test_dataset)}")

    ########################################################################
    # 1) Fit scaler on training set for inputs (and optionally for targets)
    ########################################################################
    print("Fitting StandardScaler using combined incremental and parallel approach on the training set (inputs)...")
    scaler_x = gather_all_inputs_combined(train_dataset, n_jobs=4)

    scaler_x_path = os.path.join(checkpoint_dir, f"scalerX_{input_mode}.pkl")
    joblib.dump(scaler_x, scaler_x_path)
    print(f"Input scaler saved to: {scaler_x_path}")

    if scale_y_flag:
        print("Fitting StandardScaler on the training set (targets)...")
        all_train_Y = gather_all_targets(train_dataset)
        scaler_y = StandardScaler()
        scaler_y.fit(all_train_Y)
        scaler_y_path = os.path.join(checkpoint_dir, f"scalerY_{input_mode}.pkl")
        joblib.dump(scaler_y, scaler_y_path)
        print(f"Target scaler saved to {scaler_y_path}")
    else:
        scaler_y = None

    ########################################################################
    # 2) Wrap each dataset with ScaledDataset to apply the transforms
    ########################################################################
    scaled_train_dataset = ScaledDataset(train_dataset, scaler_x, scaler_y)
    scaled_val_dataset = ScaledDataset(val_dataset, scaler_x, scaler_y)
    scaled_test_dataset = ScaledDataset(test_dataset, scaler_x, scaler_y)

    trainLoader = DataLoader(scaled_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    validLoader = DataLoader(scaled_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    testLoader = DataLoader(scaled_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    ########################################################################
    # 3) Build model, set up optimizer, criterion, trainer
    ########################################################################
    input_size = input_sizes[input_mode]
    
    # Uncomment one of the following model selections:
    model = BasicLSTM(input_size, hidden_size, num_layers, output_size, n_ahead).to(device)
    # model = TimeSeriesTransformer(input_size, num_heads=4, num_layers=2, hidden_dim=64, output_size=output_size).to(device)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    test_loss = nn.SmoothL1Loss()

    trainer = ModelTrainer(
        model=model, 
        loss=test_loss, 
        optimizer=optimizer, 
        accuracy=None, 
        model_type="Regression",
        device=device, 
        noPrint=False, 
        flatten_output=False
    )
    trainer.input_mode = input_mode
    trainer.epoch_log_file = epoch_log_file

    ########################################################################
    # 4) (Optional) Resume from checkpoint if it exists
    ########################################################################
    checkpoint_path = os.path.join(checkpoint_dir, f"model_{input_mode}.pth")
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Train
    t0 = datetime.now()
    trainer.fit(trainLoader, validLoader, epochs, start_epoch=start_epoch, checkpoint_dir=checkpoint_dir)
    t1 = datetime.now()

    ########################################################################
    # 5) Save the final trained model
    ########################################################################
    model_save_path = os.path.join(trained_dir, f"model_{input_mode}.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model saved to {model_save_path}")

    ########################################################################
    # 6) Evaluate on Test
    ########################################################################
    trainer.Test_Model(testLoader)
    print("\nInput config:", input_mode, 
          "Test Loss:", trainer.Metrics["Test Loss"], 
          "Training Time:", t1 - t0)
    
    # (Saving metrics and generating plots code omitted for brevity)
    results[input_mode] = trainer.Metrics

final_summary_file = os.path.join(log_base_dir, "final_loss_summary.txt")
with open(final_summary_file, "w") as f:
    f.write("Summary of Experiments:\n")
    for mode in results:
        test_loss_val = results[mode].get("Test Loss", "N/A")
        val_loss_val = results[mode].get("Validation Loss", "N/A")
        f.write(f"Input config: {mode}, Test Loss: {test_loss_val}, Validation Loss: {val_loss_val}\n")
        if "Test MSE" in results[mode]:
            f.write("Additional Metrics:\n")
            f.write(f"  MSE: {results[mode].get('Test MSE')}\n")
            f.write(f"  RMSE: {results[mode].get('Test RMSE')}\n")
            f.write(f"  MAE: {results[mode].get('Test MAE')}\n")
            f.write(f"  R2: {results[mode].get('Test R2')}\n")
            f.write(f"  Pearson: {results[mode].get('Test Pearson')}\n")
        if "Per Action Metrics" in results[mode]:
            f.write("Per Action Metrics:\n")
            for act, metrics in results[mode]["Per Action Metrics"].items():
                f.write(f"  Action: {act}, Metrics: {metrics}\n")
        f.write("\n")
