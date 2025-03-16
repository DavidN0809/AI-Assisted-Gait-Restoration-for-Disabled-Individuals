import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Import your dataset class
from utils.datasets import EMG_dataset
# Import your LSTM model
from models.models import BasicLSTM
# Import your trainer
from utils.Trainer import ModelTrainer

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# General hyperparameters
index = r'/data1/dnicho26/EMG_DATASET/processed/index.csv'
lag = 30
n_ahead = 10
batch_size = 1024
epochs = 1
lr = 0.00007
hidden_size = 128
num_layers = 5
output_size = 3  # Always predict 3 channels (target leg EMG)

# Define input configurations and corresponding sizes
input_configs = ["all", "emg", "acc", "gyro"]
input_sizes = {"all": 21, "emg": 3, "acc": 9, "gyro": 9}

results = {}

for input_mode in input_configs:
    print("\n========================================")
    print("Training model with input configuration:", input_mode)
    
    dataset = EMG_dataset(index, lag=lag, n_ahead=n_ahead,
                          input_leg="right", target_leg="left",
                          input_sensor=input_mode, target_sensor="emg",
                          )
    
    # Check an example window
    X, Y = dataset.__getitem__(0)
    print("Example input and target shapes:", X.shape, Y.shape)
    print(f"Dataset length {len(dataset)}")

    train_dataset, valid_dataset, test_dataset = random_split(dataset, [0.7, 0.1, 0.2])
    trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validLoader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    testLoader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)
    
    input_size = input_sizes[input_mode]
    model = BasicLSTM(input_size, hidden_size, num_layers, output_size, n_ahead).to(device)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    trainer = ModelTrainer(
        model, criterion, optimizer, None, "Regression", device, noPrint=False, flatten_output=False
    )
        
    # Ensure checkpoint directory exists
    checkpoint_dir = "/data1/dnicho26/Thesis/AI-Assisted-Gait-Restoration-for-Disabled-Individuals/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f"model_{input_mode}.pth")
    start_epoch = 0
    
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    t0 = datetime.now()
    trainer.fit(trainLoader, validLoader, epochs, start_epoch=start_epoch)
    t1 = datetime.now()
    
    torch.save({
        'epoch': epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    
    trainer.Test_Model(testLoader)
    print("\nInput config:", input_mode, "Test Loss:", trainer.Metrics["Test Loss"], "Training Time:", t1 - t0)
    trainer.Graph_Metrics(save_path=f"/data1/dnicho26/Thesis/AI-Assisted-Gait-Restoration-for-Disabled-Individuals/figures/training/metrics_{input_mode}.png")
    
    for X, Y in testLoader:
        X, Y = X.to(device), Y.to(device)
        pred = model(X).detach().cpu().numpy()
        
        y_range = range(len(X[0]), len(X[0]) + n_ahead)
        plt.figure(figsize=(8, 4))
        plt.plot(y_range, pred[0, :], 'b', label="Prediction")
        plt.plot(y_range, Y[0].detach().cpu().numpy(), 'g', label="Actual")
        plt.legend()
        plt.title(f"Sample Prediction - Input config: {input_mode}")
        plt.xlabel("Timestep")
        plt.ylabel("Signal Amplitude")
        plt.savefig(f"/data1/dnicho26/Thesis/AI-Assisted-Gait-Restoration-for-Disabled-Individuals/figures/training/sample_prediction_{input_mode}.png")
        plt.close()
        print(f"Saved sample prediction plot for {input_mode}")
        break
    
    results[input_mode] = trainer.Metrics

print("\nSummary of Experiments:")
for mode in results:
    print(f"Input config: {mode}, Test Loss: {results[mode]['Test Loss']}")

# Save test loss and validation loss to a file
log_file = "/data1/dnicho26/Thesis/AI-Assisted-Gait-Restoration-for-Disabled-Individuals/logs/loss_summary.txt"
os.makedirs(os.path.dirname(log_file), exist_ok=True)  # Ensure the logs directory exists

with open(log_file, "w") as f:
    f.write("Summary of Experiments:\n")
    for mode in results:
        test_loss = results[mode].get("Test Loss", "N/A")
        val_loss = results[mode].get("Validation Loss", "N/A")
        f.write(f"Input config: {mode}, Test Loss: {test_loss}, Validation Loss: {val_loss}\n")

print(f"Loss summary saved to {log_file}")