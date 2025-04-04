#!/usr/bin/env python3
import os
import re
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob

def custom_loss(predictions, targets, variation_weight=0.3, eps=1e-5):
    # MAPE calculation
    percentage_errors = torch.abs((targets - predictions) / (targets + eps))
    mape_loss = percentage_errors.mean()
    
    # Only compute variation loss if there is more than one forecast time step
    if predictions.shape[1] > 1:
        pred_diff = torch.abs(predictions[:, 1:, :] - predictions[:, :-1, :])
        target_diff = torch.abs(targets[:, 1:, :] - targets[:, :-1, :])
        variation_loss = torch.abs(pred_diff - target_diff).mean()
    else:
        variation_loss = 0.0

    return mape_loss + variation_weight * variation_loss

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience,
    but only considers epochs where the validation loss is below a target threshold.
    """
    def __init__(self, patience=5, min_delta=0.0, target_loss=0.01, verbose=False, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.target_loss = target_loss
        self.verbose = verbose
        self.path = path
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        # Only consider early stopping logic if the validation loss is below the target threshold.
        if val_loss > self.target_loss:
            if self.verbose:
                print(f"Validation loss {val_loss:.6f} is above target threshold {self.target_loss:.6f}. Not incrementing early stopping counter.")
            return

        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print("Validation loss decreased. Saving model ...")
        torch.save(model.state_dict(), self.path)

class Trainer:
    def __init__(
        self,
        model,
        lag,
        n_ahead,          # forecast steps per sample
        optimizer,
        scheduler,
        testloader,
        fig_dir,
        loss='custom',    # 'mse', 'smoothl1', 'huber', or 'custom'
        model_type="lstm",
        device="cpu",
        is_classification=False,
        **kwargs
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.testloader = testloader
        self.fig_dir = fig_dir
        self.model_type = model_type.lower()
        self.device = device
        self.is_classification = is_classification
        self.lag = lag
        self.n_ahead = n_ahead
        
        if loss == "mse":
            self.criterion = nn.MSELoss(reduction="none")
        elif loss == "smoothl1":
            self.criterion = nn.SmoothL1Loss(reduction="none")
        elif loss == "huber":
            self.criterion = nn.HuberLoss(delta=0.5, reduction="none")
        elif loss == "custom":
            self.criterion = custom_loss
        else:
            raise ValueError("Invalid loss type provided.")
            
        self.metrics = {"train_loss": [], "val_loss": []}
        self.model.to(self.device)

    @property
    def Metrics(self):
        return {
            "Training Loss": self.metrics["train_loss"],
            "Validation Loss": self.metrics["val_loss"],
        }

    def train_epoch(self, dataloader):
        """
        Uses input X (lag) to predict the full forecast.
        Loss is computed over the forecast sequence.
        """
        self.model.train()
        total_loss = 0
        count = 0
        for batch in tqdm(dataloader, desc="Training", leave=False):
            X, Y, _, _, weight = batch  # Y: [B, n_ahead, channels]
            X = X.to(self.device)
            Y = Y.to(self.device)
            weight = weight.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(X)  # shape: [B, n_ahead, channels]
            loss = self.criterion(predictions, Y)
            if loss.dim() > 0:
                loss = (loss.mean(dim=(1, 2)) * weight).mean()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * X.size(0)
            count += X.size(0)
        avg_loss = total_loss / count
        self.metrics["train_loss"].append(avg_loss)

    def val_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0
        count = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation", leave=False):
                X, Y, _, _, weight = batch  # weight is still loaded but not used here
                X = X.to(self.device)
                Y = Y.to(self.device)
                predictions = self.model(X)
                loss = self.criterion(predictions, Y)
                # Compute mean loss over forecast dimensions without weighting
                if loss.dim() > 0:
                    loss = loss.mean(dim=(1, 2)).mean()
                total_loss += loss.item() * X.size(0)
                count += X.size(0)
        avg_loss = total_loss / count
        self.metrics["val_loss"].append(avg_loss)

    def Test_Model(self, loader, save_path):
        self.model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in loader:
                X, Y, *rest = batch  # X: [B, lag, channels], Y: [B, n_ahead, channels]
                preds = self.model(X)
                all_preds.append(preds)
                all_targets.append(Y)
        self.test_results = {
            'preds': torch.cat(all_preds, dim=0),
            'targets': torch.cat(all_targets, dim=0)
        }
        self.plot_test_results(save_path=save_path,test_loader=loader)

    def step_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def save_first_windows(self, train_loader, epoch, num_windows=10):
        """
        Save a given number of sample forecast windows from the training set.
        """
        save_dir = os.path.join(self.fig_dir, self.model_type, f"epoch_{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving {num_windows} sample windows to: {save_dir}")
        
        self.model.eval()
        saved_count = 0

        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Saving Windows", leave=False):
                X_batch, Y_batch = batch[0], batch[1]  # X: [B, lag, channels], Y: [B, n_ahead, channels]
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)

                batch_size = X_batch.size(0)
                for i in range(batch_size):
                    if saved_count >= num_windows:
                        break

                    # Extract forecasts.
                    gt_forecast = Y_batch[i].detach().cpu().numpy()  # expected: [n_ahead, channels]
                    pred_forecast = self.model(X_batch[i].unsqueeze(0))
                    pred_forecast = pred_forecast.squeeze(0).detach().cpu().numpy()  # expected: [n_ahead, channels]

                    # If forecast dimensions are swapped, transpose.
                    if gt_forecast.shape[0] != self.n_ahead:
                        gt_forecast = gt_forecast.T
                    if pred_forecast.shape[0] != self.n_ahead:
                        pred_forecast = pred_forecast.T

                    n_channels = gt_forecast.shape[1] if gt_forecast.ndim > 1 else 1
                    fig, axs = plt.subplots(n_channels, 1, figsize=(8, 3 * n_channels))
                    if n_channels == 1:
                        axs = [axs]
                    for ch in range(n_channels):
                        axs[ch].plot(np.arange(self.n_ahead), gt_forecast[:, ch], marker='o', label=f"GT Forecast Ch{ch}")
                        axs[ch].plot(np.arange(self.n_ahead), pred_forecast[:, ch], marker='x', linestyle='--', label=f"Pred Forecast Ch{ch}")
                        axs[ch].set_xlabel("Forecast Time Step")
                        axs[ch].set_ylabel("Signal Value")
                        axs[ch].legend()
                        axs[ch].grid(True)
                    fig.suptitle(f"Epoch {epoch} - Sample {saved_count}")
                    save_path = os.path.join(save_dir, f"window_{saved_count}.png")
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    plt.savefig(save_path)
                    plt.close(fig)

                    saved_count += 1
                if saved_count >= num_windows:
                    break

    def plot_validation_results(self, val_loader, save_path, num_windows=3):
        """
        Plot validation results using only the forecast (n_ahead) portion.
        Each sample's forecast is plotted on subplots (one per channel) showing GT vs. predicted.
        """
        self.model.eval()
        batch = next(iter(val_loader))  # grab one batch
        X, Y = batch[0], batch[1]       # X: [B, lag, channels], Y: [B, n_ahead, channels]
        X = X.to(self.device)
        Y = Y.to(self.device)

        num_plots = min(num_windows, X.shape[0])
        
        with torch.no_grad():
            for i in range(num_plots):
                gt_forecast = Y[i].detach().cpu().numpy()  # expected shape: [n_ahead, channels]
                pred_forecast = self.model(X[i].unsqueeze(0)).squeeze(0).detach().cpu().numpy()  # expected shape: [n_ahead, channels]

                if gt_forecast.shape[0] != self.n_ahead:
                    gt_forecast = gt_forecast.T
                if pred_forecast.shape[0] != self.n_ahead:
                    pred_forecast = pred_forecast.T

                n_channels = gt_forecast.shape[1] if gt_forecast.ndim > 1 else 1
                fig, axs = plt.subplots(n_channels, 1, figsize=(8, 3 * n_channels))
                if n_channels == 1:
                    axs = [axs]
                for ch in range(n_channels):
                    axs[ch].plot(np.arange(self.n_ahead), gt_forecast[:, ch], marker='o', label=f"GT Forecast Ch{ch}")
                    axs[ch].plot(np.arange(self.n_ahead), pred_forecast[:, ch], marker='x', linestyle='--', label=f"Pred Forecast Ch{ch}")
                    axs[ch].set_xlabel("Forecast Time Step")
                    axs[ch].set_ylabel("Signal Value")
                    axs[ch].legend()
                    axs[ch].grid(True)
                fig.suptitle(f"Validation Sample {i}")
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                sample_save_path = save_path.replace(".png", f"_{i}.png")
                plt.savefig(sample_save_path)
                plt.close(fig)
        print(f"Saved validation results plots to {save_path}")
    
    def plot_prediction(self, X_sample, Y_sample, Y_pred, fig_path):
        """
        Plots only the forecast (n_ahead) portion for a single sample.
        Each channel gets its own subplot with ground truth and prediction.
        """
        if Y_sample.shape[0] != self.n_ahead:
            Y_sample = Y_sample.T
        if Y_pred.shape[0] != self.n_ahead:
            Y_pred = Y_pred.T
        n_channels = Y_sample.shape[1] if Y_sample.ndim > 1 else 1
        fig, axs = plt.subplots(n_channels, 1, figsize=(8, 3 * n_channels))
        if n_channels == 1:
            axs = [axs]
        for ch in range(n_channels):
            axs[ch].plot(np.arange(self.n_ahead), Y_sample[:, ch], marker='o', label=f"GT Forecast Ch{ch}")
            axs[ch].plot(np.arange(self.n_ahead), Y_pred[:, ch], marker='x', linestyle='--', label=f"Pred Forecast Ch{ch}")
            axs[ch].set_xlabel("Forecast Time Step")
            axs[ch].set_ylabel("Signal Value")
            axs[ch].legend()
            axs[ch].grid(True)
        fig.suptitle("Forecast: Ground Truth vs. Prediction")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(fig_path)
        plt.close(fig)

    def plot_loss_curve(self, save_path):
        """
        Plots and saves the loss curves for both training and validation.
        """
        epochs = range(1, len(self.metrics["train_loss"]) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.metrics["train_loss"], 'b-', label='Training Loss')
        plt.plot(epochs, self.metrics["val_loss"], 'r-', label='Validation Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Loss curve saved to {save_path}")

    def plot_test_results(self, test_loader, save_path, num_windows=3):
        """
        Computes the average test loss over the entire test set and
        plots a given number of sample forecast windows from the test set,
        displaying the average test loss in the title of each plot.
        """
        self.model.eval()
        total_loss = 0
        count = 0
        with torch.no_grad():
            for batch in test_loader:
                X_batch, Y_batch, *rest = batch
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)
                preds = self.model(X_batch)
                loss = self.criterion(preds, Y_batch)
                if loss.dim() > 0:
                    loss = (loss.mean(dim=(1,2))).mean()
                total_loss += loss.item() * X_batch.size(0)
                count += X_batch.size(0)
        avg_test_loss = total_loss / count

        # Get a batch for plotting sample windows.
        batch = next(iter(test_loader))
        X, Y = batch[0], batch[1]
        X = X.to(self.device)
        Y = Y.to(self.device)
        predictions = self.model(X)
        num_plots = min(num_windows, X.shape[0])
        for i in range(num_plots):
            gt_forecast = Y[i].detach().cpu().numpy()  # expected shape: [n_ahead, channels]
            pred_forecast = predictions[i].detach().cpu().numpy()  # expected shape: [n_ahead, channels]
            if gt_forecast.shape[0] != self.n_ahead:
                gt_forecast = gt_forecast.T
            if pred_forecast.shape[0] != self.n_ahead:
                pred_forecast = pred_forecast.T

            n_channels = gt_forecast.shape[1] if gt_forecast.ndim > 1 else 1
            fig, axs = plt.subplots(n_channels, 1, figsize=(8, 3 * n_channels))
            if n_channels == 1:
                axs = [axs]
            for ch in range(n_channels):
                axs[ch].plot(np.arange(self.n_ahead), gt_forecast[:, ch], marker='o', label=f"GT Forecast Ch{ch}")
                axs[ch].plot(np.arange(self.n_ahead), pred_forecast[:, ch], marker='x', linestyle='--', label=f"Pred Forecast Ch{ch}")
                axs[ch].set_xlabel("Forecast Time Step")
                axs[ch].set_ylabel("Signal Value")
                axs[ch].legend()
                axs[ch].grid(True)
            fig.suptitle(f"Test Sample {i} (Avg Test Loss: {avg_test_loss:.4f})")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            sample_save_path = save_path.replace(".png", f"_{i}.png")
            plt.savefig(sample_save_path)
            plt.close(fig)
        print(f"Test results saved to {save_path}")
        
    def log_epoch_loss(self, epoch, train_loss, val_loss, log_file_path):
        with open(log_file_path, "a") as f:
            f.write(f"{epoch},{train_loss},{val_loss}\n")

    def fit(self, train_loader, val_loader, epochs, checkpoint_dir, 
            patience=5, min_delta=0.0, num_windows=10, loss_curve_path='loss_curve.png'):
        """
        Runs the training and validation loops for a number of epochs,
        saving per-epoch and best checkpoints in the specified checkpoint_dir.
        If a checkpoint exists in checkpoint_dir, resume training from it.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")
        epoch_checkpoint_prefix = os.path.join(checkpoint_dir, "model_checkpoint")

        # Resume logic: check for a best checkpoint or per-epoch checkpoints.
        start_epoch = 0
        if os.path.exists(best_checkpoint_path):
            print("Best checkpoint found. Loading best checkpoint for evaluation/resume.")
            self.model.load_state_dict(torch.load(best_checkpoint_path))
            # Optionally, evaluate the model on the test set if available.
            if hasattr(self, 'testloader'):
                print("Evaluating best checkpoint on test set:")
                self.Test_Model(self.testloader)
            # Here you can decide whether to resume training from the best checkpoint.
        else:
            epoch_checkpoint_pattern = epoch_checkpoint_prefix + "_epoch_*.pth"
            epoch_checkpoint_files = glob.glob(epoch_checkpoint_pattern)
            if epoch_checkpoint_files:
                latest_checkpoint_file = max(epoch_checkpoint_files, key=lambda x: int(re.search(r'epoch_(\d+)\.pth', x).group(1)))
                checkpoint = torch.load(latest_checkpoint_file)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"Resuming training from epoch {start_epoch}")
        
        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, verbose=True, path=best_checkpoint_path)
        
        # Training loop (runs for the specified number of additional epochs)
        for epoch in range(start_epoch, start_epoch + epochs):
            print(f"\nEpoch {epoch+1}/{start_epoch + epochs}")
            
            # Training phase
            self.train_epoch(train_loader)
            train_loss = self.metrics["train_loss"][-1]
            print(f"Training Loss: {train_loss:.4f}")
            
            # Validation phase
            self.val_epoch(val_loader)
            val_loss = self.metrics["val_loss"][-1]
            print(f"Validation Loss: {val_loss:.4f}, Lag: {self.lag}")
                        
            # Log the loss values (assuming self.epoch_log_file is set)
            if hasattr(self, 'epoch_log_file') and self.epoch_log_file:
                self.log_epoch_loss(epoch+1, train_loss, val_loss, self.epoch_log_file)

            # Save per-epoch checkpoint
            epoch_checkpoint_path = f"{epoch_checkpoint_prefix}_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, epoch_checkpoint_path)
            print(f"Epoch checkpoint saved to {epoch_checkpoint_path}")
            
            # Save sample forecast windows for this epoch
            self.save_first_windows(train_loader, epoch+1, num_windows=num_windows)
            
            # Early stopping check (saves best checkpoint automatically)
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping triggered. Stopping training.")
                break
            
            # Step the scheduler if applicable
            self.step_scheduler()
        
        # After training, plot and save the loss curve
        self.plot_loss_curve(loss_curve_path)
