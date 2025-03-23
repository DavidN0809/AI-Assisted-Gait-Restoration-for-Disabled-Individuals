import os
from torchmetrics import ConfusionMatrix
from matplotlib import pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import copy
import torch
import math
from sklearn.metrics import f1_score  


class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()
        
    def forward(self, y_pred, y_true):
        # log(cosh(x)) is approximately x^2/2 for small x and |x| - log(2) for large x.
        loss = torch.log(torch.cosh(y_pred - y_true))
        return torch.mean(loss)


class EarlyStopping():
    def __init__(self, patience=5, min_delta=0.00005, restore_best_weights=True):
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
            model: The PyTorch model.
            loss: Loss function (e.g. nn.MSELoss() for regression or nn.CrossEntropyLoss() for classification).
            optimizer: Optimizer.
            accuracy: Accuracy metric function for classification, or a regression metric function.
            model_type (str): "Classification" or "Regression".
            device (str): 'cuda' or 'cpu'.
            classes (int): Number of classes (only used for classification).
            noPrint (bool): If True, suppress printing.
            flatten_output (bool): If True, flatten predictions and labels before computing loss.
        """
        self.device = device
        self.model = model.to(device)
        self.Loss_Function = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
        self.model_type = model_type,
        self.model_name = model_name  # New attribute for model type
        self.input_mode = input_mode
        self.classNum = classes
        self.noPrint = noPrint
        self.flatten_output = flatten_output
        self.testLoader = testloader
        self.fig_dir=fig_dir
        
        # For flattening outputs if needed
        if flatten_output:
            self.flat = nn.Flatten()
        
        if noPrint:
            self.progressbar = lambda x: x
        else:
            self.progressbar = tqdm
        
        # Dictionary to store metrics over epochs.
        if model_type == "Classification":
            self.Metrics = {"Training Loss": [], "Validation Loss": [],
                            "Training Accuracy": [], "Validation Accuracy": [],
                            "Test Loss": 0, "Test Accuracy": 0, "Test F1 Score": 0}
            self.ConfMatrix = None
        else:
            # For regression, also track additional metrics.
            self.Metrics = {"Training Loss": [], "Validation Loss": [],
                            "Test Loss": 0, "Test MSE": 0, "Test RMSE": 0, "Test MAE": 0,
                            "Test R2": 0, "Test Pearson": 0, "Per Action Metrics": {}}
    
    def Training_Loop(self, Loader):
        self.model.train()
        tLossSum = 0
        tAccuracy = 0
                
        for batch_idx, (data, labels, *_) in enumerate(self.progressbar(Loader)):

            # if batch_idx == 0:  # only print stats for the first batch to avoid too much console spam
            #     y_mean = labels.mean().item()
            #     y_std  = labels.std().item()
            #     y_min  = labels.min().item()
            #     y_max  = labels.max().item()
            #     print(f"[DEBUG] Y stats - mean: {y_mean}, std: {y_std}, min: {y_min}, max: {y_max}")

            #     # Flatten the batch and time dimensions, keeping channels as the last dimension.
            #     channel_means = labels.view(-1, labels.shape[-1]).mean(dim=0)
            #     channel_stds  = labels.view(-1, labels.shape[-1]).std(dim=0)
            #     channel_mins  = labels.view(-1, labels.shape[-1]).min(dim=0)[0]
            #     channel_maxs  = labels.view(-1, labels.shape[-1]).max(dim=0)[0]

            #     print(f"[DEBUG] Channel-wise Mean: {channel_means}")
            #     print(f"[DEBUG] Channel-wise STD: {channel_stds}")
            #     print(f"[DEBUG] Channel-wise Min: {channel_mins}")
            #     print(f"[DEBUG] Channel-wise Max: {channel_maxs}")


            #     # Suppose labels has shape [batch_size, sequence_length, num_channels]
            #     # Check channel-wise std:
            #     channel_std = labels.view(-1, labels.shape[-1]).std(dim=0)
            #     print(f"[DEBUG] Channel-wise STD: {channel_std}")

            #     unique_vals = torch.unique(labels)
            #     print(f"[DEBUG] Unique values in labels: {unique_vals}")

            data = data.to(self.device)
            
            if self.model_type == "Classification":
                # Convert labels to one-hot vectors.
                labels = torch.eye(self.classNum)[labels].to(self.device)
            else:
                labels = labels.to(self.device)
            
            pred = self.model(data)
            
            if self.flatten_output:
                loss_val = self.Loss_Function(self.flat(pred), self.flat(labels))
            else:
                loss_val = self.Loss_Function(pred, labels)
            
            self.optimizer.zero_grad()
            loss_val.backward()
            self.optimizer.step()
            
            tLossSum += loss_val.item()
            
            # For classification, calculate accuracy.
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
                if self.model_type == "Classification":
                    labels = torch.eye(self.classNum)[labels].to(self.device)
                else:
                    labels = labels.to(self.device)
                
                pred = self.model(data)
                if self.flatten_output:
                    loss_val = self.Loss_Function(self.flat(pred), self.flat(labels))
                else:
                    loss_val = self.Loss_Function(pred, labels)
                    
                tLossSum += loss_val.item()
                
                if self.model_type == "Classification":
                    pred_labels = torch.tensor([torch.argmax(i).item() for i in pred]).to(self.device)
                    true_labels = torch.tensor([torch.argmax(i).item() for i in labels]).to(self.device)
                    tAccuracy += self.accuracy(pred_labels, true_labels)
                    
        self.Metrics["Validation Loss"].append(tLossSum / len(Loader))
        if self.model_type == "Classification":
            self.Metrics["Validation Accuracy"].append(tAccuracy / len(Loader))
    
    def plot_predictions(self, epoch, num_windows=5):
        """Plots predictions vs. ground truth for multiple test windows per epoch.
        Displays the action and leg for each sample.

        Parameters:
        - epoch (int): The current training epoch.
        - num_windows (int): The number of test samples to plot per epoch.
        """
        self.model.eval()

        # 1) Grab a batch from the test Loader.
        test_batch = next(iter(self.testLoader))  
        X_test, Y_test, actions, legs = test_batch  # unpack all four

        # 2) Send X and Y to the device.
        X_test = X_test.to(self.device)
        Y_test = Y_test.to(self.device)

        # 3) Run inference (disable gradient since we’re just plotting).
        with torch.no_grad():
            Y_pred = self.model(X_test)

        # Ensure we don't exceed available samples
        num_windows = min(num_windows, X_test.shape[0])

        # 4) Create or ensure the output directory for epoch plots exists.
        epoch_fig_dir = os.path.join(self.fig_dir, "epoch_plots")
        os.makedirs(epoch_fig_dir, exist_ok=True)

        # 5) Plot multiple test samples
        for i in range(num_windows):
            Y_test_np = Y_test[i].cpu().numpy()  # shape: [sequence_length, num_channels]
            Y_pred_np = Y_pred[i].cpu().numpy()  # same shape
            current_action = actions[i]  # e.g. 'walking', etc.
            current_leg = legs

            plt.figure(figsize=(10, 6))
            for ch in range(Y_test_np.shape[-1]):
                plt.plot(Y_test_np[:, ch], label=f"Sensor {ch} Actual")
                plt.plot(Y_pred_np[:, ch], label=f"Sensor {ch} Predicted", linestyle="--")

            plt.xlabel("Time Step")
            plt.ylabel("Signal Value")
            plt.title(
                f"Epoch {epoch+1} - Sample {i+1} ({self.model_name})\n"
                f"Action: {current_action}, Leg: {current_leg[i]}"
            )
            plt.legend()

            # 6) Save and close the figure
            plot_path = os.path.join(epoch_fig_dir, f"epoch_{epoch+1}_gt{current_leg[i]}_sample_{i+1}.png")
            plt.savefig(plot_path)
            plt.close()

            # 7) Print a confirmation
            print(f"Saved epoch {epoch+1} sample {i+1} plot to {plot_path}")
            print(f"Action: {current_action}, Leg: {current_leg[i]}")

    def fit(self, trainingLoader, validateLoader, EPOCHS, start_epoch=0, checkpoint_dir=None):
        ES = EarlyStopping()
        for epoch in range(start_epoch, EPOCHS):
            self.Training_Loop(trainingLoader)
            self.Validation_Loop(validateLoader)
            
            if not self.noPrint:
                print("EPOCH:", epoch + 1)
                print("Training Loss:", self.Metrics["Training Loss"][-1],
                    " | Validation Loss:", self.Metrics["Validation Loss"][-1])
                if self.model_type == "Classification":
                    print("Training Accuracy:", self.Metrics["Training Accuracy"][-1],
                        " | Validation Accuracy:", self.Metrics["Validation Accuracy"][-1])
            
            # Save current epoch metrics to the log file (append mode)
            if hasattr(self, "epoch_log_file"):
                with open(self.epoch_log_file, "a") as f:
                    f.write(f"Epoch: {epoch+1}\n")
                    f.write("Training Loss: {}\n".format(self.Metrics["Training Loss"][-1]))
                    f.write("Validation Loss: {}\n".format(self.Metrics["Validation Loss"][-1]))
                    if self.model_type == "Classification":
                        f.write("Training Accuracy: {}\n".format(self.Metrics["Training Accuracy"][-1]))
                        f.write("Validation Accuracy: {}\n".format(self.Metrics["Validation Accuracy"][-1]))
                    f.write("\n")

            # **Save checkpoint every epoch**
            if checkpoint_dir is not None:
                checkpoint_path = os.path.join(checkpoint_dir, f"model_{self.input_mode}_{self.model_name}_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
             
            self.plot_predictions(epoch)
            
            # Check early stopping condition
            if ES(self.model, self.Metrics["Validation Loss"][-1]):
                if not self.noPrint:
                    print("Stopping Model Early:", ES.status)
                break

    def Test_Model(self, testLoader):
        self.model.eval()
        total_loss = 0
        
        if self.model_type == "Classification":
            # For classification tasks, accumulate predicted and true labels.
            all_pred_labels = []
            all_true_labels = []
            
            with torch.no_grad():
                for batch in testLoader:
                    if isinstance(batch, (list, tuple)) and len(batch) == 3:
                        data, labels, _ = batch
                    else:
                        data, labels = batch
                        
                    data = data.to(self.device)
                    # Convert labels to one-hot vectors for loss calculation.
                    labels_onehot = torch.eye(self.classNum)[labels].to(self.device)
                    pred = self.model(data)
                    if self.flatten_output:
                        loss_val = self.Loss_Function(self.flat(pred), self.flat(labels_onehot))
                    else:
                        loss_val = self.Loss_Function(pred, labels_onehot)
                        
                    total_loss += loss_val.item()
                    
                    # Get predicted labels from model output.
                    pred_labels = torch.argmax(pred, dim=1)
                    # For true labels, use the original integer labels.
                    all_pred_labels.extend(pred_labels.cpu().numpy().tolist())
                    all_true_labels.extend(labels.cpu().numpy().tolist())
            
            self.Metrics["Test Loss"] = total_loss / len(testLoader)
            # Calculate accuracy using the provided accuracy function.
            acc = self.accuracy(torch.tensor(all_pred_labels), torch.tensor(all_true_labels))
            self.Metrics["Test Accuracy"] = acc.item() if isinstance(acc, torch.Tensor) else acc
            # Calculate F1 score (weighted)
            f1 = f1_score(all_true_labels, all_pred_labels, average='weighted')
            self.Metrics["Test F1 Score"] = f1
            
            self.test_results = {"preds": all_pred_labels, "targets": all_true_labels}
            
        else:
            # For regression tasks.
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
                        loss_val = self.Loss_Function(self.flat(pred), self.flat(labels))
                    else:
                        loss_val = self.Loss_Function(pred, labels)
                        
                    total_loss += loss_val.item()
                    
                    for i in range(data.size(0)):
                        all_preds.append(pred[i].cpu())
                        all_targets.append(labels[i].cpu())
                        if isinstance(actions, list):
                            all_actions.append(actions[i])
                        else:
                            all_actions.append(actions[i])
            
            # Stack all predictions and targets into tensors.
            all_preds_tensor = torch.stack(all_preds)
            all_targets_tensor = torch.stack(all_targets)
            
            # Standard regression metrics.
            mse = ((all_preds_tensor - all_targets_tensor)**2).mean().item()
            rmse = math.sqrt(mse)
            mae = torch.abs(all_preds_tensor - all_targets_tensor).mean().item()
            
            self.Metrics["Test Loss"] = total_loss / len(testLoader)
            self.Metrics["Test MSE"] = mse
            self.Metrics["Test RMSE"] = rmse
            self.Metrics["Test MAE"] = mae
            
            # Compute R-squared.
            ss_res = ((all_targets_tensor - all_preds_tensor)**2).sum()
            ss_tot = ((all_targets_tensor - all_targets_tensor.mean())**2).sum()
            r2 = 1 - ss_res/ss_tot
            self.Metrics["Test R2"] = r2.item()
            
            # Compute Pearson correlation coefficient.
            y_true_flat = all_targets_tensor.view(-1)
            y_pred_flat = all_preds_tensor.view(-1)
            cov = ((y_true_flat - y_true_flat.mean()) * (y_pred_flat - y_pred_flat.mean())).sum()
            std_true = torch.sqrt(((y_true_flat - y_true_flat.mean())**2).sum())
            std_pred = torch.sqrt(((y_pred_flat - y_pred_flat.mean())**2).sum())
            pearson = cov / (std_true * std_pred)
            self.Metrics["Test Pearson"] = pearson.item()
            
            # Compute per-action metrics.
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
