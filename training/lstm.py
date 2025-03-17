import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
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
index = r"D:\UNC Charlotte Dropbox\orgs-ecgr-QuantitativeImagingandAILaboratory/processed/index.csv"

#base_dir="/data1/dnicho26/Thesis/AI-Assisted-Gait-Restoration-for-Disabled-Individuals"
base_dir = r"C:\Users\alway\OneDrive\Documents\GitHub\AI-Assisted-Gait-Restoration-for-Disabled-Individuals"

lag = 30
n_ahead = 10
batch_size = 512
epochs = 300
lr = 1e-7
hidden_size = 128
num_layers = 5
output_size = 3  # Always predict 3 channels (target leg EMG)

# Ensure checkpoint directory exists
checkpoint_dir = f"{base_dir}/models/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Ensure trained models directory exists
trained_dir = f"{base_dir}/models/trained"
os.makedirs(trained_dir, exist_ok=True)

# Define input configurations and corresponding sizes
input_configs = ["all", "emg", "acc", "gyro"]
input_sizes = {"all": 21, "emg": 3, "acc": 9, "gyro": 9}

results = {}

log_base_dir = f"{base_dir}/logs"
os.makedirs(log_base_dir, exist_ok=True)

fig_base_dir = f"{base_dir}/figures/training"
os.makedirs(fig_base_dir, exist_ok=True)

results = {}

for input_mode in input_configs:
    print("\n========================================")
    print("Training model with input configuration:", input_mode)
    
    # Create a model-specific log file (this will be appended each epoch)
    epoch_log_file = os.path.join(log_base_dir, f"loss_summary_{input_mode}.txt")
    # Optionally, clear previous log:
    with open(epoch_log_file, "w") as f:
        f.write(f"Loss summary for input config: {input_mode}\n\n")
    
    # Create a model-specific figures directory:
    fig_dir = os.path.join(fig_base_dir, input_mode)
    os.makedirs(fig_dir, exist_ok=True)
    
    dataset = EMG_dataset(index, lag=lag, n_ahead=n_ahead,
                          input_sensor=input_mode, target_sensor="emg")
    
    # Check an example window (now returns action as well)
    X, Y, action = dataset.__getitem__(0)
    print("Example input and target shapes:", X.shape, Y.shape, "Action:", action)
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
    # Pass the current input_mode to the trainer so it can use it in saving logs
    trainer.input_mode = input_mode  
    trainer.epoch_log_file = epoch_log_file  # add a reference to the log file path

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
    
    model_save_path = os.path.join(trained_dir, f"model_{input_mode}.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model saved to {model_save_path}")

    trainer.Test_Model(testLoader)
    print("\nInput config:", input_mode, "Test Loss:", trainer.Metrics["Test Loss"], "Training Time:", t1 - t0)
    
    metrics_all_file = os.path.join(log_base_dir, f"metrics_all_{input_mode}.txt")

    # Save the complete metrics for the current model
    with open(metrics_all_file, "w") as f:
        f.write(f"Metrics for input configuration: {input_mode}\n")
        for metric_name, metric_value in trainer.Metrics.items():
            f.write(f"{metric_name}: {metric_value}\n")
    print(f"Saved complete metrics to {metrics_all_file}")

    # Save training metrics figure in the model-specific folder.
    metrics_fig_path = os.path.join(fig_dir, f"metrics_{input_mode}.png")
    trainer.Graph_Metrics(save_path=metrics_fig_path)
    
        # Define mapping for left leg muscles.
    left_mapping = {
        0: "Gastrocnemius",
        1: "Biceps Femoris",
        2: "Quadriceps Femoris"
    }

    # Graph sample predictions for each action.
    test_results = trainer.test_results
    action_to_sample = {}
    for pred, target, act in zip(test_results["preds"], test_results["targets"], test_results["actions"]):
        if act not in action_to_sample:
            action_to_sample[act] = (pred, target)

    for act, (pred, target) in action_to_sample.items():
        y_range = range(pred.shape[0])
        fig, axs = plt.subplots(1, pred.shape[1], figsize=(6*pred.shape[1], 4))
        if pred.shape[1] == 1:
            axs = [axs]
        for ch in range(pred.shape[1]):
            axs[ch].plot(y_range, pred[:, ch].numpy(), 'b', marker='o', 
                       label=f"{left_mapping.get(ch, 'Channel ' + str(ch))} Prediction")
            axs[ch].plot(y_range, target[:, ch].numpy(), 'g', marker='o', 
                       label=f"{left_mapping.get(ch, 'Channel ' + str(ch))} Actual")
            axs[ch].set_title(left_mapping.get(ch, f"Channel {ch}"))
            axs[ch].set_xlabel("Timestep")
            axs[ch].set_ylabel("Signal Amplitude")
            axs[ch].legend()
        plt.suptitle(f"Sample Muscles Prediction - Action: {act}")
        sample_pred_path = os.path.join(fig_dir, f"prediction_{input_mode}_{act}.png")
        plt.savefig(sample_pred_path)
        plt.close()
        print(f"Saved sample prediction plot for action: {act} with input config {input_mode}")
    
    # Additionally, generate a "sample_prediction_all" plot.
    # for batch in testLoader:
    #     if isinstance(batch, (list, tuple)) and len(batch) == 3:
    #         data, labels, _ = batch
    #     else:
    #         data, labels = batch
    #     data = data.to(device)
    #     labels = labels.to(device)
    #     pred = model(data).detach().cpu()
    #     labels = labels.cpu()
    #     sample_pred = pred[0]  # shape: [n_ahead, channels]
    #     sample_target = labels[0]
    #     y_range = range(sample_pred.shape[0])
        
    #     fig, axs = plt.subplots(1, sample_pred.shape[1], figsize=(6*sample_pred.shape[1], 4))
    #     if sample_pred.shape[1] == 1:
    #         axs = [axs]
    #     for ch in range(sample_pred.shape[1]):
    #         axs[ch].plot(y_range, sample_pred[:, ch].numpy(), 'b', marker='o', 
    #                    label=f"{left_mapping.get(ch, 'Channel ' + str(ch))} Prediction")
    #         axs[ch].plot(y_range, sample_target[:, ch].numpy(), 'g', marker='o', 
    #                    label=f"{left_mapping.get(ch, 'Channel ' + str(ch))} Actual")
    #         axs[ch].set_title(left_mapping.get(ch, f"Channel {ch}"))
    #         axs[ch].set_xlabel("Timestep")
    #         axs[ch].set_ylabel("Signal Amplitude")
    #         axs[ch].legend()
    #     plt.suptitle("Sample Muscles Prediction - All")
    #     sample_pred_all_path = os.path.join(fig_dir, f"sample_prediction_all_{input_mode}.png")
    #     plt.savefig(sample_pred_all_path)
    #     plt.close()
    #     print(f"Saved sample prediction all plot for input config {input_mode}")
    #     break

    results[input_mode] = trainer.Metrics

final_summary_file = os.path.join(log_base_dir, "final_loss_summary.txt")
with open(final_summary_file, "w") as f:
    f.write("Summary of Experiments:\n")
    for mode in results:
        test_loss = results[mode].get("Test Loss", "N/A")
        val_loss = results[mode].get("Validation Loss", "N/A")
        f.write(f"Input config: {mode}, Test Loss: {test_loss}, Validation Loss: {val_loss}\n")
        # For regression, include additional metrics
        if "Test MSE" in results[mode]:
            f.write("Additional Metrics:\n")
            f.write(f"  MSE: {results[mode].get('Test MSE')}\n")
            f.write(f"  RMSE: {results[mode].get('Test RMSE')}\n")
            f.write(f"  MAE: {results[mode].get('Test MAE')}\n")
            f.write(f"  R2: {results[mode].get('Test R2')}\n")
            f.write(f"  Pearson: {results[mode].get('Test Pearson')}\n")
        # For classification, you might log different metrics
        if "Per Action Metrics" in results[mode]:
            f.write("Per Action Metrics:\n")
            for act, metrics in results[mode]["Per Action Metrics"].items():
                f.write(f"  Action: {act}, Metrics: {metrics}\n")
        f.write("\n")
