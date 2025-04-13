#!/usr/bin/env python
"""
Inference script for a saved model checkpoint.
This script:
 • Loads the saved scaler for inputs.
 • Loads the test dataset.
 • Instantiates the model (e.g. BasicLSTM) with the proper hyperparameters.
 • Loads the saved model weights from a checkpoint.
 • Processes one sample from the test dataset (applying the fitted scaler).
 • Runs inference and plots the predicted vs. actual EMG signals.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
import numpy as np
import matplotlib.pyplot as plt
import joblib
from torch.utils.data import DataLoader

# Import your dataset and model classes
from utils.datasets import EMG_dataset
from models.models import BasicLSTM  # change as needed if using another model variant
from utils.plot_styles import plot_prediction_vs_ground_truth

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------------
# Configuration & Paths
# -------------------------
# Paths to saved model and scaler (update these paths as needed)
trained_model_path = "/data1/dnicho26/Thesis/AI-Assisted-Gait-Restoration-for-Disabled-Individuals/runs/run2/models/checkpoints/model_all_BasicLSTM_10.pth"
# trained_model_path = "/data1/dnicho26/Thesis/AI-Assisted-Gait-Restoration-for-Disabled-Individuals/runs/run2/models/trained/model_all_BasicLSTM_4.pt"
scaler_x_path      = "/data1/dnicho26/Thesis/AI-Assisted-Gait-Restoration-for-Disabled-Individuals/runs/run2/models/checkpoints/scalerX_all.pkl"

# Test dataset CSV index path
test_index  = "/data1/dnicho26/EMG_DATASET/processed-server/index_testtreadmill.csv"

# Input sensor configuration (must match training)
input_mode = "all"  
input_sizes = {"all": 21, "emg": 3, "acc": 9, "gyro": 9}
input_size = input_sizes[input_mode]

# Model hyperparameters (should match training configuration)
hidden_size = 128
num_layers  = 5
output_size = 3   # Predicting 3 EMG channels
n_ahead     = 10
lag         = 30

# -------------------------
# Load the saved scaler
# -------------------------
print("Loading input scaler from:", scaler_x_path)
scaler_x = joblib.load(scaler_x_path)

# -------------------------
# Load the test dataset
# -------------------------
print("Loading Test Dataset...")
test_dataset = EMG_dataset(test_index, lag=lag, n_ahead=n_ahead,
                           input_sensor=input_mode, target_sensor="emg", randomize_legs=False)

# -------------------------
# Instantiate the model and load checkpoint
# -------------------------
print("Instantiating model and loading weights...")
model = BasicLSTM(input_size, hidden_size, num_layers, output_size, n_ahead).to(device)

import torch
from collections import OrderedDict

# Load the checkpoint
checkpoint = torch.load(trained_model_path, map_location=device)
state_dict = checkpoint["model_state_dict"]

# Remove "module." prefix from keys if present
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_key = k.replace("module.", "")
    new_state_dict[new_key] = v

# Load the modified state_dict into your model
model.load_state_dict(new_state_dict)

model.eval()  # set model to evaluation mode

# -------------------------
# Inference on a test sample
# -------------------------
# Select a sample (here, the first sample in the test dataset)
sample_idx = 0
sample_X, sample_Y, action = test_dataset[sample_idx]

# Convert sample input to numpy, scale it, and convert back to tensor
sample_X_np = sample_X.numpy()
shape = sample_X_np.shape  # expected shape: (lag, features)
sample_X_2d = sample_X_np.reshape(-1, shape[-1])
sample_X_scaled_np = scaler_x.transform(sample_X_2d)
sample_X_scaled = torch.from_numpy(sample_X_scaled_np.reshape(shape)).float().to(device)

# Add batch dimension
sample_X_scaled = sample_X_scaled.unsqueeze(0)

# Run the model (inference)
with torch.no_grad():
    pred = model(sample_X_scaled)
pred = pred.squeeze(0).cpu().numpy()  # shape: (n_ahead, output_size)

# Convert ground truth to numpy array
true = sample_Y.numpy()

# -------------------------
# Plotting: Prediction vs Ground Truth
# -------------------------
for ch in range(output_size):
    plt.figure(figsize=(10, 6))
    plot_prediction_vs_ground_truth(
        pred[:, ch],
        true[:, ch],
        title=f"Prediction vs Ground Truth - Sensor {ch}"
    )
    plt.savefig(f"/data1/dnicho26/Thesis/AI-Assisted-Gait-Restoration-for-Disabled-Individuals/inference_sensor_{ch}.png")
    plt.close()

# Create a combined plot for all sensors
plt.figure(figsize=(15, 10))
for ch in range(output_size):
    plt.subplot(output_size, 1, ch + 1)
    plot_prediction_vs_ground_truth(
        pred[:, ch],
        true[:, ch],
        title=f"Sensor {ch}"
    )
plt.tight_layout()
plt.savefig("/data1/dnicho26/Thesis/AI-Assisted-Gait-Restoration-for-Disabled-Individuals/inference_combined.png")
plt.close()
