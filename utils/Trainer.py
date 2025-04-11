import os
import re
import glob
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def custom_loss(predictions, targets, variation_weight=0.3, eps=1e-5):
    # MAPE calculation
    percentage_errors = torch.abs((targets - predictions) / (targets + eps))
    mape_loss = percentage_errors.mean()
    
    # Only compute variation loss if more than one forecast time step exists.
    if predictions.shape[1] > 1:
        pred_diff = torch.abs(predictions[:, 1:, :] - predictions[:, :-1, :])
        target_diff = torch.abs(targets[:, 1:, :] - targets[:, :-1, :])
        variation_loss = torch.abs(pred_diff - target_diff).mean()
    else:
        variation_loss = 0.0

    return mape_loss + variation_weight * variation_loss

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=10, min_delta=0.0, verbose=False, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.path = path
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            if self.verbose:
                print(f"Validation loss decreased to {val_loss:.6f}. Saving best model and resetting counter.")
        else:
            self.counter += 1
            if self.verbose:
                print(f"Validation loss did not improve. EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered. Stopping training.")

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Saving best model checkpoint with validation loss: {val_loss:.6f}")
        torch.save(model.state_dict(), self.path)

class Trainer:
    def __init__(self, model, lag, n_ahead, optimizer, scheduler, testloader, fig_dir,
                 loss='custom', model_type="lstm", device="cpu", is_classification=False, **kwargs):
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
        return {"Training Loss": self.metrics["train_loss"], "Validation Loss": self.metrics["val_loss"]}
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        count = 0
        for batch in tqdm(dataloader, desc="Training", leave=False):
            X, Y, _, weight, *rest = batch  # Y: [B, n_ahead, channels]
            X = X.to(self.device)
            Y = Y.to(self.device)
            if isinstance(weight, list):
                weight = torch.tensor(weight)
            weight = weight.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(X)
            loss = self.criterion(predictions, Y)
            if loss.dim() > 0:
                loss = loss.mean(dim=(1,2)).mean()
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
                X, Y, _, weight = batch
                X = X.to(self.device)
                Y = Y.to(self.device)
                predictions = self.model(X)
                loss = self.criterion(predictions, Y)
                if loss.dim() > 0:
                    loss = loss.mean(dim=(1,2)).mean()
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
                X, Y, *rest = batch
                preds = self.model(X)
                all_preds.append(preds)
                all_targets.append(Y)
        self.test_results = {
            'preds': torch.cat(all_preds, dim=0),
            'targets': torch.cat(all_targets, dim=0)
        }
        self.plot_test_results(test_loader=loader, save_path=save_path)
    
    def step_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()
    
    def log_epoch_loss(self, epoch, train_loss, val_loss, log_file_path):
        with open(log_file_path, "a") as f:
            f.write(f"{epoch},{train_loss},{val_loss}\n")
    
    def fit(self, train_loader, val_loader, epochs, checkpoint_dir, patience=5,
            min_delta=0.0, num_windows=10, loss_curve_path='loss_curve.png'):
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")
        epoch_checkpoint_prefix = os.path.join(checkpoint_dir, "model_checkpoint")
        
        # If a best checkpoint already exists, assume training is done.
        if os.path.exists(best_checkpoint_path):
            print("Best checkpoint found. Skipping training as model is already trained.")
            self.model.load_state_dict(torch.load(best_checkpoint_path))
            if hasattr(self, 'testloader'):
                best_test_save_path = os.path.join(self.fig_dir, "best_test_results.png")
                print("Evaluating best checkpoint on test set:")
                self.Test_Model(self.testloader, best_test_save_path)
            return
        
        start_epoch = 0
        # Check for epoch checkpoints to resume training.
        epoch_checkpoint_pattern = epoch_checkpoint_prefix + "_checkpoint_epoch_*.pth"
        epoch_checkpoint_files = glob.glob(epoch_checkpoint_pattern)
        if epoch_checkpoint_files:
            latest_checkpoint_file = max(epoch_checkpoint_files, key=lambda x: int(re.search(r'epoch_(\d+)\.pth', x).group(1)))
            checkpoint = torch.load(latest_checkpoint_file)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}")
        
        # early_stopping = EarlyStopping(patience=patience, verbose=True, path=best_checkpoint_path)
        
        for epoch in range(start_epoch, start_epoch + epochs):
            print(f"\nEpoch {epoch+1}/{start_epoch + epochs}")
            
            self.train_epoch(train_loader)
            train_loss = self.metrics["train_loss"][-1]
            print(f"Training Loss: {train_loss:.4f}")
            
            self.val_epoch(val_loader)
            val_loss = self.metrics["val_loss"][-1]
            print(f"Validation Loss: {val_loss:.4f}, Lag: {self.lag}")
            
            if hasattr(self, 'epoch_log_file') and self.epoch_log_file:
                self.log_epoch_loss(epoch+1, train_loss, val_loss, self.epoch_log_file)
            
            # Save epoch checkpoint for resuming.
            epoch_checkpoint_path = f"{epoch_checkpoint_prefix}_checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, epoch_checkpoint_path)
            print(f"Epoch checkpoint saved to {epoch_checkpoint_path}")
            
            self.save_first_windows(train_loader, epoch+1, num_windows=num_windows)
            
            # early_stopping(val_loss, self.model)
            # if early_stopping.early_stop:
            #     print("Early stopping triggered. Training stopped early.")
            #     break
            
            self.step_scheduler()
        
        # If no best checkpoint was saved during training (i.e. training finished all epochs), save final model.
        if not os.path.exists(best_checkpoint_path):
            print("Training complete. Saving final model as best checkpoint.")
            torch.save(self.model.state_dict(), best_checkpoint_path)
        
        self.plot_loss_curve(loss_curve_path)
    
    def plot_prediction(self, X_sample, Y_sample, Y_pred, fig_path):
        if Y_sample.shape[0] != self.n_ahead:
            Y_sample = Y_sample.T
        if Y_pred.shape[0] != self.n_ahead:
            Y_pred = Y_pred.T
        
        lag_input_np = X_sample.detach().cpu().numpy() if isinstance(X_sample, torch.Tensor) else X_sample
        n_lag = lag_input_np.shape[0]
        # Use forecast's channel count.
        n_channels = Y_sample.shape[1] if Y_sample.ndim > 1 else 1
        
        x_lag = np.arange(-n_lag, 0)
        x_forecast = np.arange(self.n_ahead)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        for ch in range(n_channels):
            ax.plot(x_lag, lag_input_np[:, ch], marker='o', label=f"Lag Sensor {ch}")
            ax.plot(x_forecast, Y_sample[:, ch], marker='o', label=f"GT Forecast Ch{ch}")
            ax.plot(x_forecast, Y_pred[:, ch], marker='x', linestyle='--', label=f"Pred Forecast Ch{ch}")
        
        ax.axvline(x=0, color='gray', linestyle=':', linewidth=1)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Signal Value")
        ax.set_title("Lag Input & Forecast: Ground Truth vs. Prediction")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close(fig)
    
    def plot_validation_results(self, val_loader, save_path, num_windows=3):
        self.model.eval()
        batch = next(iter(val_loader))
        X, Y = batch[0], batch[1]
        X = X.to(self.device)
        Y = Y.to(self.device)
        num_plots = min(num_windows, X.shape[0])
        
        with torch.no_grad():
            for i in range(num_plots):
                lag_input_np = X[i].detach().cpu().numpy()
                gt_forecast = Y[i].detach().cpu().numpy()
                pred_forecast = self.model(X[i].unsqueeze(0)).squeeze(0).detach().cpu().numpy()
                
                if gt_forecast.shape[0] != self.n_ahead:
                    gt_forecast = gt_forecast.T
                if pred_forecast.shape[0] != self.n_ahead:
                    pred_forecast = pred_forecast.T
                
                n_lag = lag_input_np.shape[0]
                x_lag = np.arange(-n_lag, 0)
                x_forecast = np.arange(self.n_ahead)
                # Use the number of forecast channels.
                n_channels = gt_forecast.shape[1] if gt_forecast.ndim > 1 else 1
                
                fig, ax = plt.subplots(figsize=(8, 6))
                for ch in range(n_channels):
                    ax.plot(x_lag, lag_input_np[:, ch], marker='o', label=f"Lag Sensor {ch}")
                    ax.plot(x_forecast, gt_forecast[:, ch], marker='o', label=f"GT Forecast Ch{ch}")
                    ax.plot(x_forecast, pred_forecast[:, ch], marker='x', linestyle='--', label=f"Pred Forecast Ch{ch}")
                
                ax.axvline(x=0, color='gray', linestyle=':', linewidth=1)
                ax.set_xlabel("Time Step")
                ax.set_ylabel("Signal Value")
                ax.set_title(f"Validation Sample {i}")
                ax.legend()
                ax.grid(True)
                plt.tight_layout()
                sample_save_path = save_path.replace(".png", f"_{i}.png")
                plt.savefig(sample_save_path)
                plt.close(fig)
        print(f"Saved validation results plots to {save_path}")
    
    def plot_test_results(self, test_loader, save_path, num_windows=3):
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
                    loss = loss.mean(dim=(1,2)).mean()
                total_loss += loss.item() * X_batch.size(0)
                count += X_batch.size(0)
        avg_test_loss = total_loss / count

        batch = next(iter(test_loader))
        X, Y = batch[0], batch[1]
        X = X.to(self.device)
        Y = Y.to(self.device)
        predictions = self.model(X)
        num_plots = min(num_windows, X.shape[0])
        for i in range(num_plots):
            lag_input_np = X[i].detach().cpu().numpy()
            gt_forecast = Y[i].detach().cpu().numpy()
            pred_forecast = predictions[i].detach().cpu().numpy()
            if gt_forecast.shape[0] != self.n_ahead:
                gt_forecast = gt_forecast.T
            if pred_forecast.shape[0] != self.n_ahead:
                pred_forecast = pred_forecast.T
            
            n_lag = lag_input_np.shape[0]
            x_lag = np.arange(-n_lag, 0)
            x_forecast = np.arange(self.n_ahead)
            # Use forecast's channel dimension for plotting.
            n_channels = gt_forecast.shape[1] if gt_forecast.ndim > 1 else 1
            
            fig, ax = plt.subplots(figsize=(8, 6))
            for ch in range(n_channels):
                ax.plot(x_lag, lag_input_np[:, ch], marker='o', label=f"Lag Sensor {ch}")
                ax.plot(x_forecast, gt_forecast[:, ch], marker='o', label=f"GT Forecast Ch{ch}")
                ax.plot(x_forecast, pred_forecast[:, ch], marker='x', linestyle='--', label=f"Pred Forecast Ch{ch}")
            
            ax.axvline(x=0, color='gray', linestyle=':', linewidth=1)
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Signal Value")
            ax.set_title(f"Test Sample {i} (Avg Test Loss: {avg_test_loss:.4f})")
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            sample_save_path = save_path.replace(".png", f"_{i}.png")
            plt.savefig(sample_save_path)
            plt.close(fig)
        print(f"Test results saved to {save_path}")
    
    def plot_loss_curve(self, save_path):
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
    
    def save_first_windows(self, train_loader, epoch, num_windows=5):
        save_dir = os.path.join(self.fig_dir, self.model_type, f"epoch_{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving sample windows to: {save_dir}")
        
        self.model.eval()
        action_counts = {}
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Saving Windows", leave=False):
                X_batch = batch[0]
                Y_batch = batch[1]
                actions = batch[2]
                batch_size = Y_batch.size(0)
                for i in range(batch_size):
                    action = actions[i].item() if isinstance(actions[i], torch.Tensor) else actions[i]
                    if action_counts.get(action, 0) >= num_windows:
                        continue
                    lag_input_np = X_batch[i].detach().cpu().numpy()
                    gt_forecast = Y_batch[i].detach().cpu().numpy()
                    sample_input = X_batch[i].unsqueeze(0)
                    pred_forecast = self.model(sample_input).squeeze(0).detach().cpu().numpy()
                    
                    if gt_forecast.shape[0] != self.n_ahead:
                        gt_forecast = gt_forecast.T
                    if pred_forecast.shape[0] != self.n_ahead:
                        pred_forecast = pred_forecast.T
                    
                    n_lag = lag_input_np.shape[0]
                    n_channels_lag = lag_input_np.shape[1] if lag_input_np.ndim > 1 else 1
                    n_channels_forecast = gt_forecast.shape[1] if gt_forecast.ndim > 1 else 1
                    
                    x_lag = np.arange(-n_lag, 0)
                    x_forecast = np.arange(self.n_ahead)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    for ch in range(n_channels_lag):
                        ax.plot(x_lag, lag_input_np[:, ch], marker='o', label=f"Lag Sensor {ch}")
                    for ch in range(n_channels_forecast):
                        ax.plot(x_forecast, gt_forecast[:, ch], marker='o', label=f"GT Sensor {ch}")
                        ax.plot(x_forecast, pred_forecast[:, ch], marker='x', linestyle='--', label=f"Pred Sensor {ch}")
                    
                    ax.axvline(x=0, color='gray', linestyle=':', linewidth=1)
                    ax.set_xlabel("Time Step")
                    ax.set_ylabel("Signal Value")
                    ax.set_title(f"Epoch {epoch} - Action {action} - Sample {action_counts.get(action, 0)+1}")
                    ax.legend(fontsize=8)
                    ax.grid(True)
                    
                    plt.tight_layout()
                    file_name = f"action_{action}_epoch_{epoch}_sample_{action_counts.get(action, 0)+1}.png"
                    save_path_fig = os.path.join(save_dir, file_name)
                    plt.savefig(save_path_fig)
                    plt.close(fig)
                    
                    action_counts[action] = action_counts.get(action, 0) + 1
                    
                if action_counts and all(count >= num_windows for count in action_counts.values()):
                    break
