import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from datetime import datetime
import matplotlib.pyplot as plt

# Import all model variants from your models file.
from models.models import BasicLSTM, BidirectionalLSTM, ResidualLSTM, AttentionLSTM
from utils.datasets import EMG_dataset
from utils.Trainer import ModelTrainer

# Use GPU if available.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Define paths and hyperparameters.
index = r"D:\UNC Charlotte Dropbox\orgs-ecgr-QuantitativeImagingandAILaboratory/processed/index.csv"
base_dir = r"C:\Users\alway\OneDrive\Documents\GitHub\AI-Assisted-Gait-Restoration-for-Disabled-Individuals"

lag = 30
n_ahead = 10
batch_size = 64
epochs = 300
lr = 7e-7  # Increased learning rate.
hidden_size = 256
num_layers = 5
output_size = 3  # Always predict 3 channels (target leg EMG).
input_config = "all"  # Using all sensor inputs.

# Directories for checkpoints, trained models, logs, and figures.
checkpoint_dir = os.path.join(base_dir, "models", "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
trained_dir = os.path.join(base_dir, "models", "trained")
os.makedirs(trained_dir, exist_ok=True)
log_base_dir = os.path.join(base_dir, "logs")
os.makedirs(log_base_dir, exist_ok=True)
fig_base_dir = os.path.join(base_dir, "figures", "training")
os.makedirs(fig_base_dir, exist_ok=True)

# Prepare dataset using all sensor inputs.
dataset = EMG_dataset(index, lag=lag, n_ahead=n_ahead,
                      input_sensor=input_config, target_sensor="emg", randomize_legs=False)
print("Dataset length:", len(dataset))
train_dataset, valid_dataset, test_dataset = random_split(dataset, [0.7, 0.1, 0.2])
trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
validLoader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
testLoader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Define input sizes mapping.
input_sizes = {"all": 21, "emg": 3, "acc": 9, "gyro": 9}
input_size = input_sizes[input_config]

# Define the model variants to train.
model_variants = {
    'AttentionLSTM': AttentionLSTM,
    'ResidualLSTM': ResidualLSTM,
    'BidirectionalLSTM': BidirectionalLSTM,
    'BasicLSTM': BasicLSTM
}

results = {}

for variant_name, ModelClass in model_variants.items():
    print("\n========================================")
    print("Training model variant:", variant_name)
    
    # Create log file and figures directory for this variant.
    epoch_log_file = os.path.join(log_base_dir, f"loss_summary_{variant_name}.txt")
    with open(epoch_log_file, "w") as f:
        f.write(f"Loss summary for model variant: {variant_name}\n\n")
    fig_dir = os.path.join(fig_base_dir, variant_name)
    os.makedirs(fig_dir, exist_ok=True)
    
    # Initialize the model variant.
    model = ModelClass(input_size, hidden_size, num_layers, output_size, n_ahead).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    # Create an optimizer with the increased learning rate.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # Use SmoothL1Loss (Huber loss) for regression.
    criterion = nn.SmoothL1Loss()
    
    # Create the trainer.
    trainer = ModelTrainer(
        model, criterion, optimizer, None, "Regression", device, noPrint=False, flatten_output=False
    )
    trainer.epoch_log_file = epoch_log_file
    
    # Set up checkpoint path for this variant.
    checkpoint_path = os.path.join(checkpoint_dir, f"model_{variant_name}.pth")
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Train the model.
    t0 = datetime.now()
    trainer.fit(trainLoader, validLoader, epochs, start_epoch=start_epoch)
    t1 = datetime.now()
    
    # Save checkpoint and trained model.
    torch.save({
        'epoch': epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    
    model_save_path = os.path.join(trained_dir, f"model_{variant_name}.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model saved to {model_save_path}")
    
    # Evaluate the model on the test set.
    trainer.Test_Model(testLoader)
    print("\nModel variant:", variant_name, "Test Loss:", trainer.Metrics["Test Loss"], "Training Time:", t1 - t0)
    
    # Save metrics and figures.
    metrics_all_file = os.path.join(log_base_dir, f"metrics_all_{variant_name}.txt")
    with open(metrics_all_file, "w") as f:
        f.write(f"Metrics for model variant: {variant_name}\n")
        for metric_name, metric_value in trainer.Metrics.items():
            f.write(f"{metric_name}: {metric_value}\n")
    print(f"Saved complete metrics to {metrics_all_file}")
    
    metrics_fig_path = os.path.join(fig_dir, f"metrics_{variant_name}.png")
    trainer.Graph_Metrics(save_path=metrics_fig_path)
    
    results[variant_name] = trainer.Metrics

# Save a final summary for all variants.
final_summary_file = os.path.join(log_base_dir, "final_loss_summary.txt")
with open(final_summary_file, "w") as f:
    f.write("Summary of Experiments (Model Variants):\n")
    for variant in results:
        test_loss = results[variant].get("Test Loss", "N/A")
        val_loss = results[variant].get("Validation Loss", "N/A")
        f.write(f"Model Variant: {variant}, Test Loss: {test_loss}, Validation Loss: {val_loss}\n")
        if "Test MSE" in results[variant]:
            f.write("Additional Metrics:\n")
            f.write(f"  MSE: {results[variant].get('Test MSE')}\n")
            f.write(f"  RMSE: {results[variant].get('Test RMSE')}\n")
            f.write(f"  MAE: {results[variant].get('Test MAE')}\n")
            f.write(f"  R2: {results[variant].get('Test R2')}\n")
            f.write(f"  Pearson: {results[variant].get('Test Pearson')}\n")
        if "Per Action Metrics" in results[variant]:
            f.write("Per Action Metrics:\n")
            for act, metrics in results[variant]["Per Action Metrics"].items():
                f.write(f"  Action: {act}, Metrics: {metrics}\n")
        f.write("\n")
print("Final summary saved to", final_summary_file)
