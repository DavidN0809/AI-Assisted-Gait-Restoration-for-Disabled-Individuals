import os
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import ConfusionMatrix
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np


#############################################
# Loss Functions
#############################################

class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()
        
    def forward(self, y_pred, y_true):
        # log(cosh(x)) approximates x^2/2 for small x and |x| - log(2) for large x.
        loss = torch.log(torch.cosh(y_pred - y_true))
        return torch.mean(loss)

class QuantileLoss(nn.Module):
    """
    Computes the quantile loss for a given quantile.
    For quantile q, the loss is:
        L = max(q*(y_true - y_pred), (q-1)*(y_true - y_pred))
    """
    def __init__(self, quantile):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile

    def forward(self, y_pred, y_true):
        error = y_true - y_pred
        loss = torch.max(self.quantile * error, (self.quantile - 1) * error)
        return torch.mean(loss)

class CustomEMGLoss(nn.Module):
    """
    Computes per-channel loss for multi-channel EMG signals.

    This version always uses QuantileLoss as its base loss.
    Additionally, it adds an auxiliary input loss (chosen via input_loss_type)
    and extra terms to encourage dynamic signal behavior:
      - Derivative loss: compares differences between successive time steps.
      - Correlation loss: 1 minus the Pearson correlation between prediction and target.
      - Forecast start loss: penalizes the difference between the first predicted value and the first ground truth forecast value.

    The final loss is computed as:
      Total Loss = QuantileLoss(pred,true)
                   + input_loss_weight * aux_loss(pred,true)
                   + derivative_weight * MSE(diff(pred), diff(true))
                   + correlation_weight * (1 - correlation(pred,true))
                   + forecast_start_weight * MSE(pred[:,0,:], true[:,0,:])
    """
    def __init__(self, sensor_indices, quantile=0.9, input_loss_type="logcosh", input_loss_weight=0.5,
                 derivative_weight=0.1, correlation_weight=0.1, forecast_start_weight=0.1):
        super(CustomEMGLoss, self).__init__()
        self.sensor_indices = sensor_indices
        # Base loss is always QuantileLoss.
        self.base_loss_fn = QuantileLoss(quantile)
        self.quantile = quantile
        
        # Auxiliary loss selection:
        self.input_loss_type = input_loss_type.lower()
        self.input_loss_weight = input_loss_weight
        self.derivative_weight = derivative_weight
        self.correlation_weight = correlation_weight
        self.forecast_start_weight = forecast_start_weight
        
        if self.input_loss_type == "logcosh":
            self.aux_loss_fn = LogCoshLoss()
        elif self.input_loss_type == "huber":
            self.aux_loss_fn = nn.HuberLoss()
        elif self.input_loss_type == "none":
            self.aux_loss_fn = None
        elif self.input_loss_type == "smoothl1loss":
            self.aux_loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError("Unsupported input_loss_type. Choose 'logcosh', 'huber', or 'none'.")

    def forward(self, y_pred, y_true):
        per_sensor_losses = {}

        # Base quantile loss
        base_loss_val = self.base_loss_fn(y_pred, y_true)

        # Auxiliary loss
        if self.aux_loss_fn is not None:
            aux_loss_val = self.aux_loss_fn(y_pred, y_true)
        else:
            aux_loss_val = 0.0

        # Derivative loss: comparing differences between successive timesteps
        derivative_loss_val = 0.0
        if y_pred.size(1) > 1:
            diff_pred = y_pred[:, 1:] - y_pred[:, :-1]
            diff_true = y_true[:, 1:] - y_true[:, :-1]
            derivative_loss_val = F.mse_loss(diff_pred, diff_true)

        # Correlation loss: compute per sensor and average over sensors
        corr_losses = []
        for idx in self.sensor_indices:
            pred_channel = y_pred[..., idx]
            true_channel = y_true[..., idx]
            mean_pred = torch.mean(pred_channel, dim=1, keepdim=True)
            mean_true = torch.mean(true_channel, dim=1, keepdim=True)
            cov = torch.mean((pred_channel - mean_pred) * (true_channel - mean_true), dim=1)
            std_pred = torch.std(pred_channel, dim=1)
            std_true = torch.std(true_channel, dim=1)
            eps = 1e-8
            corr = cov / (std_pred * std_true + eps)
            corr_losses.append(1 - torch.mean(corr))
        correlation_loss_val = torch.mean(torch.stack(corr_losses)) if len(corr_losses) > 0 else 0.0

        # Total loss from main components
        total_loss = base_loss_val \
                     + self.input_loss_weight * aux_loss_val \
                     + self.derivative_weight * derivative_loss_val \
                     + self.correlation_weight * correlation_loss_val

        # Forecast start loss: comparing the first predicted timestep with the first ground truth forecast value
        forecast_start_loss = F.mse_loss(y_pred[:, 0, :], y_true[:, 0, :])
        total_loss += self.forecast_start_weight * forecast_start_loss

        # For logging per sensor (using the same loss for each sensor)
        for idx in self.sensor_indices:
            per_sensor_losses[idx] = total_loss.detach().cpu().item()

        return total_loss, per_sensor_losses


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""
        
    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model.load_state_dict(model.state_dict())
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.status = f"Stopped on {self.counter}"
            if self.restore_best_weights:
                model.load_state_dict(self.best_model.state_dict())
            return True
        
        self.status = f"{self.counter}/{self.patience}"
        return False

class ModelTrainer:
    def __init__(self, model, loss, optimizer, accuracy, model_type, model_name, input_mode, testloader, fig_dir,
                 device, classes=0, noPrint=False, flatten_output=False):
        """
        Args:
            model: PyTorch model.
            loss: Loss function (if it returns a tuple, the first element is used).
            optimizer: Optimizer.
            accuracy: Accuracy function (for classification) or regression metric.
            model_type (str): "Classification" or "Regression".
            device (str): 'cuda' or 'cpu'.
            classes (int): Number of classes (for classification).
            noPrint (bool): If True, suppresses printing.
            flatten_output (bool): If True, flattens predictions and labels before computing loss.
            testloader: DataLoader for test data.
            fig_dir: Directory to save figures.
        """
        self.device = device
        self.model = model.to(device)
        self.Loss_Function = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
        self.model_type = model_type
        self.model_name = model_name
        self.input_mode = input_mode
        self.classNum = classes
        self.noPrint = noPrint
        self.flatten_output = flatten_output
        self.testLoader = testloader
        self.fig_dir = fig_dir
        
        if flatten_output:
            self.flat = nn.Flatten()
        
        self.progressbar = (lambda x: x) if noPrint else tqdm
        
        if model_type == "Classification":
            self.Metrics = {"Training Loss": [], "Validation Loss": [],
                            "Training Accuracy": [], "Validation Accuracy": [],
                            "Test Loss": 0, "Test Accuracy": 0, "Test F1 Score": 0}
            self.ConfMatrix = None
        else:
            self.Metrics = {"Training Loss": [], "Validation Loss": [],
                            "Test Loss": 0, "Test MSE": 0, "Test RMSE": 0, "Test MAE": 0,
                            "Test R2": 0, "Test Pearson": 0, "Per Action Metrics": {}}
    
    def Training_Loop(self, Loader):
        self.model.train()
        tLossSum = 0
        tAccuracy = 0
                
        for batch_idx, (data, labels, *_) in enumerate(self.progressbar(Loader)):
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            pred = self.model(data)
            
            if self.flatten_output:
                output = self.flat(pred)
                target = self.flat(labels)
            else:
                output = pred
                target = labels

            loss_result = self.Loss_Function(output, target)
            if isinstance(loss_result, tuple):
                loss_val = loss_result[0]
            else:
                loss_val = loss_result
            
            self.optimizer.zero_grad()
            loss_val.backward()
            self.optimizer.step()
            
            tLossSum += loss_val.item()
            
            if self.model_type == "Classification":
                pred_labels = torch.tensor([torch.argmax(i).item() for i in pred]).to(self.device)
                true_labels = torch.tensor([torch.argmax(i).item() for i in labels]).to(self.device)
                tAccuracy += self.accuracy(pred_labels, true_labels)
        
        self.Metrics["Training Loss"].append(tLossSum / len(Loader))
        if self.model_type == "Classification":
            self.Metrics["Training Accuracy"].append(tAccuracy / len(Loader))
    
    def Validation_Loop(self, Loader):
        self.model.eval()
        tLossSum = 0
        tAccuracy = 0
        
        with torch.no_grad():
            for batch in Loader:
                if isinstance(batch, (list, tuple)):
                    data, labels = batch[0], batch[1]
                else:
                    data, labels = batch
                    
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                pred = self.model(data)
                if self.flatten_output:
                    output = self.flat(pred)
                    target = self.flat(labels)
                else:
                    output = pred
                    target = labels

                loss_result = self.Loss_Function(output, target)
                if isinstance(loss_result, tuple):
                    loss_val = loss_result[0]
                else:
                    loss_val = loss_result
                    
                tLossSum += loss_val.item()
                
                if self.model_type == "Classification":
                    pred_labels = torch.tensor([torch.argmax(i).item() for i in pred]).to(self.device)
                    true_labels = torch.tensor([torch.argmax(i).item() for i in labels]).to(self.device)
                    tAccuracy += self.accuracy(pred_labels, true_labels)
                    
        self.Metrics["Validation Loss"].append(tLossSum / len(Loader))
        if self.model_type == "Classification":
            self.Metrics["Validation Accuracy"].append(tAccuracy / len(Loader))
    

    def plot_predictions(self, epoch, num_windows=10, skip_windows=30):
        """
        For each of num_windows samples from the test batch (skipping the first skip_windows),
        creates a figure with three subplots (one per EMG sensor). The x-axis covers the input (lag)
        and forecast (n_ahead) time steps. A vertical dashed line is drawn at the forecast boundary.
        
        The plots are saved to:
        ./figures/training/{model_name}/epoch_{epoch_number}/window_{sample_number}.png
        """
        self.model.eval()
        # Grab one batch from test data
        test_batch = next(iter(self.testLoader))
        X_test, Y_test, actions, target_legs = test_batch
        X_test = X_test.to(self.device)
        Y_test = Y_test.to(self.device)
        
        with torch.no_grad():
            Y_pred = self.model(X_test)
        
        # Create folder for the current epoch's figures
        base_dir = os.path.join(self.fig_dir, "training", self.model_name, f"epoch_{epoch+1}")
        os.makedirs(base_dir, exist_ok=True)
        
        # Loop starting at the skip index, then plot the next num_windows samples
        for sample_idx in range(skip_windows, skip_windows + num_windows):
            # Get sample from the batch
            X_sample = X_test[sample_idx].detach().cpu().numpy()    # shape: (lag, features)
            Y_sample = Y_test[sample_idx].detach().cpu().numpy()      # shape: (n_ahead, features)
            Y_pred_sample = Y_pred[sample_idx].detach().cpu().numpy() # shape: (n_ahead, features)
            
            # Create a figure with 3 subplots (one for each EMG sensor)
            fig, axs = plt.subplots(3, 1, figsize=(10, 12))
            
            # Total steps = input (lag) + forecast (n_ahead)
            total_steps = X_sample.shape[0] + Y_sample.shape[0]
            x_axis = list(range(total_steps))
            
            for ch in range(3):
                # Get the input (lag) for channel ch
                input_signal = X_sample[:, ch]
                # Get the ground truth forecast and predicted forecast for channel ch
                true_forecast = Y_sample[:, ch]
                pred_forecast = Y_pred_sample[:, ch]
                
                # Optionally, plot the input and forecast segments separately for clarity:
                axs[ch].plot(range(len(input_signal)), input_signal, label="Input", marker="o")
                axs[ch].plot(range(len(input_signal), total_steps), true_forecast, label="Ground Truth", marker="o")
                axs[ch].plot(range(len(input_signal), total_steps), pred_forecast, label="Predicted", marker="x", linestyle="--")
                
                # Draw a vertical dashed line at the forecast boundary (end of input)
                axs[ch].axvline(x=len(input_signal)-0.5, color="black", linestyle="--", label="Forecast Boundary")
                
                axs[ch].set_title(f"EMG Sensor {ch} - Action: {actions[sample_idx]}, Target Leg: {target_legs[sample_idx]}")
                axs[ch].set_xlabel("Time Step")
                axs[ch].set_ylabel("Signal Value")
                axs[ch].legend()
            
            plt.tight_layout()
            save_path = os.path.join(base_dir, f"window_{sample_idx+1}.png")
            plt.savefig(save_path)
            plt.close()

    def fit(self, trainingLoader, validateLoader, epochs, start_epoch=0, checkpoint_dir=None):
        ES = EarlyStopping()
        for epoch in range(start_epoch, epochs):
            self.Training_Loop(trainingLoader)
            self.Validation_Loop(validateLoader)
            
            if not self.noPrint:
                print("EPOCH:", epoch + 1)
                print("Training Loss:", self.Metrics["Training Loss"][-1],
                      "| Validation Loss:", self.Metrics["Validation Loss"][-1])
                if self.model_type == "Classification":
                    print("Training Accuracy:", self.Metrics["Training Accuracy"][-1],
                          "| Validation Accuracy:", self.Metrics["Validation Accuracy"][-1])
            
            if hasattr(self, "epoch_log_file"):
                with open(self.epoch_log_file, "a") as f:
                    f.write(f"Epoch: {epoch+1}\n")
                    f.write("Training Loss: {}\n".format(self.Metrics["Training Loss"][-1]))
                    f.write("Validation Loss: {}\n".format(self.Metrics["Validation Loss"][-1]))
                    if self.model_type == "Classification":
                        f.write("Training Accuracy: {}\n".format(self.Metrics["Training Accuracy"][-1]))
                        f.write("Validation Accuracy: {}\n".format(self.Metrics["Validation Accuracy"][-1]))
                    f.write("\n")
            
            if checkpoint_dir is not None:
                checkpoint_path = os.path.join(checkpoint_dir, f"model_{self.input_mode}_{self.model_name}_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
             
            self.plot_predictions(epoch)
            
            if ES(self.model, self.Metrics["Validation Loss"][-1]):
                if not self.noPrint:
                    print("Stopping Model Early:", ES.status)
                break

    def Test_Model(self, testLoader):
        self.model.eval()
        total_loss = 0
        
        if self.model_type == "Classification":
            all_pred_labels = []
            all_true_labels = []
            with torch.no_grad():
                for batch in testLoader:
                    if isinstance(batch, (list, tuple)) and len(batch) == 3:
                        data, labels, _ = batch
                    else:
                        data, labels = batch
                    data = data.to(self.device)
                    labels_onehot = torch.eye(self.classNum)[labels].to(self.device)
                    pred = self.model(data)
                    if self.flatten_output:
                        output = self.flat(pred)
                        target = self.flat(labels_onehot)
                    else:
                        output = pred
                        target = labels_onehot
                    loss_result = self.Loss_Function(output, target)
                    if isinstance(loss_result, tuple):
                        loss_val = loss_result[0]
                    else:
                        loss_val = loss_result
                    total_loss += loss_val.item()
                    pred_labels = torch.argmax(pred, dim=1)
                    all_pred_labels.extend(pred_labels.cpu().numpy().tolist())
                    all_true_labels.extend(labels.cpu().numpy().tolist())
            
            self.Metrics["Test Loss"] = total_loss / len(testLoader)
            acc = self.accuracy(torch.tensor(all_pred_labels), torch.tensor(all_true_labels))
            self.Metrics["Test Accuracy"] = acc.item() if isinstance(acc, torch.Tensor) else acc
            f1 = f1_score(all_true_labels, all_pred_labels, average='weighted')
            self.Metrics["Test F1 Score"] = f1
            self.test_results = {"preds": all_pred_labels, "targets": all_true_labels}
            
        else:
            all_preds = []
            all_targets = []
            all_actions = []
            with torch.no_grad():
                for batch in testLoader:
                    if isinstance(batch, (list, tuple)):
                        data, labels, actions = batch[0], batch[1], batch[2]
                    else:
                        data, labels, *_ = batch
                        actions = ["unknown"] * data.size(0)
                    data = data.to(self.device)
                    labels = labels.to(self.device)
                    pred = self.model(data)
                    if self.flatten_output:
                        output = self.flat(pred)
                        target = self.flat(labels)
                    else:
                        output = pred
                        target = labels
                    loss_result = self.Loss_Function(output, target)
                    if isinstance(loss_result, tuple):
                        loss_val = loss_result[0]
                    else:
                        loss_val = loss_result
                    total_loss += loss_val.item()
                    for i in range(data.size(0)):
                        all_preds.append(pred[i].cpu())
                        all_targets.append(labels[i].cpu())
                        if isinstance(actions, list):
                            all_actions.append(actions[i])
                        else:
                            all_actions.append(actions[i])
            
            all_preds_tensor = torch.stack(all_preds)
            all_targets_tensor = torch.stack(all_targets)
            mse = ((all_preds_tensor - all_targets_tensor)**2).mean().item()
            rmse = math.sqrt(mse)
            mae = torch.abs(all_preds_tensor - all_targets_tensor).mean().item()
            
            self.Metrics["Test Loss"] = total_loss / len(testLoader)
            self.Metrics["Test MSE"] = mse
            self.Metrics["Test RMSE"] = rmse
            self.Metrics["Test MAE"] = mae
            
            ss_res = ((all_targets_tensor - all_preds_tensor)**2).sum()
            ss_tot = ((all_targets_tensor - all_targets_tensor.mean())**2).sum()
            r2 = 1 - ss_res/ss_tot
            self.Metrics["Test R2"] = r2.item()
            
            y_true_flat = all_targets_tensor.view(-1)
            y_pred_flat = all_preds_tensor.view(-1)
            cov = ((y_true_flat - y_true_flat.mean()) * (y_pred_flat - y_pred_flat.mean())).sum()
            std_true = torch.sqrt(((y_true_flat - y_true_flat.mean())**2).sum())
            std_pred = torch.sqrt(((y_pred_flat - y_pred_flat.mean())**2).sum())
            pearson = cov / (std_true * std_pred)
            self.Metrics["Test Pearson"] = pearson.item()
            
            per_action_metrics = {}
            unique_actions = set(all_actions)
            for action in unique_actions:
                indices = [i for i, a in enumerate(all_actions) if a == action]
                if indices:
                    preds_action = torch.stack([all_preds[i] for i in indices])
                    targets_action = torch.stack([all_targets[i] for i in indices])
                    mse_action = ((preds_action - targets_action)**2).mean().item()
                    rmse_action = math.sqrt(mse_action)
                    mae_action = torch.abs(preds_action - targets_action).mean().item()
                    per_action_metrics[action] = {"MSE": mse_action, "RMSE": rmse_action, "MAE": mae_action}
            
            self.Metrics["Per Action Metrics"] = per_action_metrics
            self.test_results = {"preds": all_preds, "targets": all_targets, "actions": all_actions}

    def Graph_Metrics(self, save_path=None):
        if self.model_type == "Classification":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.plot(self.Metrics["Training Loss"], label="Training Loss", marker='o')
            ax1.plot(self.Metrics["Validation Loss"], label="Validation Loss", marker='o')
            ax1.set_title("Loss Across Epochs")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend()
            
            ax2.plot(self.Metrics["Training Accuracy"], label="Training Accuracy", marker='o')
            ax2.plot(self.Metrics["Validation Accuracy"], label="Validation Accuracy", marker='o')
            ax2.set_title("Accuracy Across Epochs")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")
            ax2.legend()
            
            if self.ConfMatrix is not None:
                plt.figure(figsize=(5, 5))
                plt.imshow(self.ConfMatrix, cmap='Blues')
                plt.title("Confusion Matrix for Test Data")
                plt.xlabel("Predicted Labels")
                plt.ylabel("True Labels")
                for i in range(self.ConfMatrix.shape[0]):
                    for j in range(self.ConfMatrix.shape[1]):
                        plt.text(j, i, self.ConfMatrix[i, j].item(), ha='center', va='center', color='black')
        else:
            fig = plt.figure(figsize=(6, 4))
            plt.plot(self.Metrics["Training Loss"], label="Training Loss", marker='o')
            plt.plot(self.Metrics["Validation Loss"], label="Validation Loss", marker='o')
            plt.title("Loss Across Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
        
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def reset(self):
        if self.model_type == "Classification":
            self.Metrics = {"Training Loss": [], "Validation Loss": [],
                            "Training Accuracy": [], "Validation Accuracy": [],
                            "Test Loss": 0, "Test Accuracy": 0, "Test F1 Score": 0}
            self.ConfMatrix = None
        else:
            self.Metrics = {"Training Loss": [], "Validation Loss": [],
                            "Test Loss": 0, "Test MSE": 0, "Test RMSE": 0, "Test MAE": 0,
                            "Test R2": 0, "Test Pearson": 0, "Per Action Metrics": {}}
            
    def plot_validation_results(self, val_loader, save_path):
        self.model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                data, labels, *_ = batch
                data = data.to(self.device)
                preds = self.model(data)
                all_preds.append(preds.cpu())
                all_targets.append(labels.cpu())
        
        # Concatenate all batches
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        
        # For simplicity, let’s assume you want to plot the first channel of all samples
        plt.figure(figsize=(12, 6))
        plt.plot(all_targets[:, :, 0].flatten(), label="Ground Truth", marker="o")
        plt.plot(all_preds[:, :, 0].flatten(), label="Predictions", marker="x", linestyle="--")
        plt.title("Validation Ground Truth vs Predictions (Channel 0)")
        plt.xlabel("Time Step (aggregated over samples)")
        plt.ylabel("Signal Value")
        plt.legend()
        plt.savefig(save_path)
        plt.close()
