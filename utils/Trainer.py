import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Trainer:
    def __init__(
        self,
        model,
        lag,
        n_ahead,          # number of forecast steps provided per sample
        optimizer,
        scheduler,
        testloader,
        fig_dir,
        loss,
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
        self.n_ahead = n_ahead  # forecast steps
        if loss == "mse":
            self.loss = nn.MSELoss(reduction="none")
        elif loss == "smoothl1":
            self.loss = nn.SmoothL1Loss(reduction="none")
        elif loss == "huber":
            self.loss = nn.HuberLoss(delta=0.5, reduction="none")
        else:
            raise ValueError("Invalid loss type provided.")
        
        self.criterion = self.loss        
        if self.is_classification:
            self.metrics = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        else:
            self.metrics = {"train_loss": [], "val_loss": []}
        self.model.to(self.device)

    @property
    def Metrics(self):
        if self.is_classification:
            return {
                "Training Loss": self.metrics["train_loss"],
                "Validation Loss": self.metrics["val_loss"],
                "Training Acc": self.metrics["train_acc"],
                "Validation Acc": self.metrics["val_acc"],
            }
        else:
            return {
                "Training Loss": self.metrics["train_loss"],
                "Validation Loss": self.metrics["val_loss"],
            }

    def train_epoch(self, dataloader):
        """
        Use X (the lag window) to predict the entire forecast sequence.
        Loss is computed comparing the model's output to Y (the full forecast).
        """
        self.model.train()
        total_loss = 0
        count = 0
        for batch in tqdm(dataloader, desc="Training", leave=False):
            X, Y, _, _, weight = batch  # Y: [B, n_ahead, channels]
            X = X.to(self.device)        # [B, lag, in_channels]
            Y = Y.to(self.device)        # [B, n_ahead, out_channels]
            weight = weight.to(self.device)

            self.optimizer.zero_grad()

            # Model outputs the full forecast sequence
            predictions = self.model(X)  # [B, n_ahead, channels]
            losses = self.criterion(predictions, Y)
            loss_per_sample = losses.mean(dim=(1, 2))
            weighted_loss = (loss_per_sample * weight).mean()

            weighted_loss.backward()
            self.optimizer.step()

            total_loss += loss_per_sample.mean().item() * X.size(0)
            count += X.size(0)

        avg_loss = total_loss / count
        self.metrics["train_loss"].append(avg_loss)
        
    def val_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0
        count = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation", leave=False):
                X, Y, _, _, weight = batch
                X = X.to(self.device)
                Y = Y.to(self.device)
                predictions = self.model(X)  # [B, n_ahead, channels]
                losses = self.criterion(predictions, Y)
                loss = losses.mean(dim=(1,2))
                total_loss += loss.sum().item()
                count += X.size(0)
        avg_loss = total_loss / count
        self.metrics["val_loss"].append(avg_loss)

    def Test_Model(self, loader):
        self.model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in loader:
                X, Y, *rest = batch  # X: [B, lag, channels], Y: [B, n_ahead, channels]
                preds = self.model(X)  # [B, n_ahead, channels]
                all_preds.append(preds)
                all_targets.append(Y)
        self.test_results = {
            'preds': torch.cat(all_preds, dim=0),
            'targets': torch.cat(all_targets, dim=0)
        }

    def Training_Loop(self, dataloader):
        self.train_epoch(dataloader)

    def Validation_Loop(self, dataloader):
        self.val_epoch(dataloader)

    def step_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def save_first_10_windows(self, train_loader, epoch):
        """
        Plot 10 examples from the training set.
        For each sample, plot:
          - The input (lag) sequence.
          - The ground truth forecast (n_ahead) and the model's prediction.
        """
        save_dir = os.path.join(self.fig_dir, self.model_type, f"epoch_{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving sample windows to: {save_dir}")
        
        self.model.eval()
        saved_count = 0

        with torch.no_grad():
            for batch in train_loader:
                X_batch, Y_batch = batch[0], batch[1]  # X: [B, lag, channels], Y: [B, n_ahead, channels]
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)

                batch_size = X_batch.size(0)
                for i in range(batch_size):
                    if saved_count >= 10:
                        break

                    # Get input and forecast parts
                    input_sequence = X_batch[i].detach().cpu().numpy()      # shape: [lag, channels]
                    gt_forecast = Y_batch[i].detach().cpu().numpy()           # shape: [n_ahead, channels]
                    
                    # Direct prediction for the forecast
                    pred_forecast = self.model(X_batch[i].unsqueeze(0))
                    pred_forecast = pred_forecast.squeeze(0).detach().cpu().numpy()  # shape: [n_ahead, channels]

                    time_input = np.arange(self.lag)
                    time_forecast = np.arange(self.lag, self.lag + self.n_ahead)
                    
                    plt.figure(figsize=(10, 6))
                    n_channels = min(3, input_sequence.shape[1])
                    for c in range(n_channels):
                        # Plot input window
                        plt.plot(time_input, input_sequence[:, c],
                                 marker='o', linestyle='-', label=f"Input Ch{c}" if c == 0 else None)
                        # Plot ground truth forecast and prediction
                        plt.plot(time_forecast, gt_forecast[:, c],
                                 marker='o', linestyle='-', label=f"GT Forecast Ch{c}" if c == 0 else None)
                        plt.plot(time_forecast, pred_forecast[:, c],
                                 marker='x', linestyle='--', label=f"Pred Forecast Ch{c}" if c == 0 else None)
                    
                    plt.axvline(x=self.lag - 0.5, color="black", linestyle="--", label="Forecast Boundary")
                    plt.title(f"Epoch {epoch} - Sample {saved_count}")
                    plt.xlabel("Time Step")
                    plt.ylabel("Signal Value")
                    plt.legend()
                    save_path = os.path.join(save_dir, f"window_{saved_count}.png")
                    plt.savefig(save_path)
                    plt.close()

                    saved_count += 1
                if saved_count >= 10:
                    break

    def plot_validation_results(self, val_loader, save_path, num_windows=3):
        """
        Plots validation results using the full forecast prediction.
        Plots the input (lag) and then the ground truth forecast versus the predicted forecast.
        """
        self.model.eval()
        batch = next(iter(val_loader))  # grab the first batch
        X, Y = batch[0], batch[1]       # X: [B, lag, channels], Y: [B, n_ahead, channels]

        X = X.to(self.device)
        Y = Y.to(self.device)

        X_cpu = X.detach().cpu().numpy()
        Y_cpu = Y.detach().cpu().numpy()

        num_plots = min(num_windows, X_cpu.shape[0])
        plt.figure(figsize=(12, 4 * num_plots))

        with torch.no_grad():
            for i in range(num_plots):
                plt.subplot(num_plots, 1, i + 1)
                input_seq = X[i].detach().cpu().numpy()    # [lag, channels]
                gt_forecast = Y[i].detach().cpu().numpy()    # [n_ahead, channels]
                pred_forecast = self.model(X[i].unsqueeze(0)).squeeze(0).detach().cpu().numpy()  # [n_ahead, channels]

                time_input = np.arange(self.lag)
                time_forecast = np.arange(self.lag, self.lag + self.n_ahead)

                n_channels = min(3, input_seq.shape[1])
                for ch in range(n_channels):
                    plt.plot(time_input, input_seq[:, ch],
                             marker='o', linestyle='-', label=f"Input Ch{ch}" if ch == 0 else None)
                    plt.plot(time_forecast, gt_forecast[:, ch],
                             marker='o', linestyle='-', label=f"GT Forecast Ch{ch}" if ch == 0 else None)
                    plt.plot(time_forecast, pred_forecast[:, ch],
                             marker='x', linestyle='--', label=f"Pred Forecast Ch{ch}" if ch == 0 else None)
                plt.axvline(x=self.lag - 0.5, color='black', linestyle='--',
                            label="Forecast Boundary" if i == 0 else None)
                plt.title(f"Validation Window {i}")
                plt.xlabel("Time Step")
                plt.ylabel("Signal Value")
                if i == 0:
                    plt.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved validation results plot to {save_path}")

    def plot_prediction(self, X_sample, Y_sample, Y_pred, fig_path):
        """
        Plots the input lag (as context) then the ground truth forecast versus predicted forecast.
        """
        plt.figure(figsize=(10, 6))
        # Plot input window
        time_input = np.arange(self.lag)
        plt.plot(time_input, X_sample[-self.lag:, 0],
                 marker='o', linestyle='-', label="Input (Lag) Ch0")
        
        # Forecast period
        time_forecast = np.arange(self.lag, self.lag + len(Y_sample))
        plt.plot(time_forecast, Y_sample[:, 0],
                 marker='o', linestyle='-', label="GT Forecast Ch0")
        plt.plot(time_forecast, Y_pred[:, 0],
                 marker='x', linestyle='--', label="Pred Forecast Ch0")
        
        plt.axvline(x=self.lag - 0.5, color="black", linestyle="--", label="Forecast Boundary")
        plt.xlabel("Time Step")
        plt.ylabel("Signal Value")
        plt.title("Lag Window, Ground Truth Forecast vs. Prediction")
        plt.legend()
        plt.savefig(fig_path)
        plt.close()
