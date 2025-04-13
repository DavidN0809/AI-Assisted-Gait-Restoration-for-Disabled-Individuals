#!/usr/bin/env python3
import os
import re
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from sklearn.metrics import r2_score
from .plot_styles import (
    PREDICTION_COLOR, GROUND_TRUTH_COLOR, PREDICTION_LINESTYLE, 
    GROUND_TRUTH_LINESTYLE, PREDICTION_LABEL, GROUND_TRUTH_LABEL, 
    get_sensor_style, get_channel_color
)

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
    but only considers epochs where the validation loss is below a target threshold
    and after the training and validation losses have converged.
    """
    def __init__(self, patience=5, min_delta=0.0, target_loss=0.01, convergence_threshold=0.1, 
                 convergence_epochs=3, verbose=False, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.target_loss = target_loss
        self.convergence_threshold = convergence_threshold  # Relative difference threshold for convergence
        self.convergence_epochs = convergence_epochs  # Number of epochs needed below threshold
        self.verbose = verbose
        self.path = path
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.convergence_counter = 0
        self.converged = False

    def check_convergence(self, train_loss, val_loss):
        """Check if training and validation losses have converged"""
        if train_loss == 0 or val_loss == 0:
            return False
            
        relative_diff = abs(train_loss - val_loss) / max(train_loss, val_loss)
        
        if relative_diff <= self.convergence_threshold:
            self.convergence_counter += 1
            if self.verbose:
                print(f"Convergence counter: {self.convergence_counter}/{self.convergence_epochs}")
            if self.convergence_counter >= self.convergence_epochs:
                if not self.converged and self.verbose:
                    print("Losses have converged. Starting early stopping monitoring.")
                self.converged = True
        else:
            self.convergence_counter = 0
            
        return self.converged

    def __call__(self, val_loss, train_loss, model):
        # First check convergence
        if not self.check_convergence(train_loss, val_loss):
            if self.verbose:
                print("Waiting for loss convergence before monitoring early stopping.")
            return
            
        # Only consider early stopping logic if the validation loss is below the target threshold
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
        clip_grad_norm=0,
        sensor_mode="emg",  # Add sensor_mode parameter with default value
        epoch_log_file=None,
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
        self.clip_grad_norm = clip_grad_norm
        self.sensor_mode = sensor_mode  # Store the sensor mode
        self.epoch_log_file = epoch_log_file  # Save the log file path
        
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
            # Handle different batch structures
            if len(batch) == 6:  # Full batch with time data (X, Y, X_time, Y_time, action, weight)
                X, Y, X_time, Y_time, _, weight = batch
                has_time_data = True
            elif len(batch) == 5:  # Batch with features (X, Y, features, action, weight)
                X, Y, _, _, weight = batch
                has_time_data = False
                X_time, Y_time = None, None
            elif len(batch) == 4:  # Batch with action and weight (X, Y, action, weight)
                X, Y, _, weight = batch
                has_time_data = False
                X_time, Y_time = None, None
            else:  # Minimal batch (X, Y, weight) or (X, Y)
                if len(batch) == 3:
                    X, Y, weight = batch
                else:
                    X, Y = batch
                    # Create a default weight tensor of ones
                    weight = torch.ones(X.size(0), device=self.device)
                has_time_data = False
                X_time, Y_time = None, None
        
            X = X.to(self.device)
            Y = Y.to(self.device)
            weight = weight.to(self.device)
        
            if X_time is not None:
                X_time = X_time.to(self.device)
            if Y_time is not None:
                Y_time = Y_time.to(self.device)

            self.optimizer.zero_grad()
        
            # Handle different model types
            if self.model_type == "informer":
                # Create dummy time features if needed
                if X_time is None:
                    X_time = torch.zeros((X.size(0), X.size(1), 5), device=self.device).long()
            
                # Configure decoder input
                label_len = self.lag // 2
                dec_inp = torch.zeros((X.size(0), self.n_ahead, X.size(2)), device=self.device)
                if label_len > 0:
                    dec_inp = torch.cat([Y[:, :label_len, :], dec_inp], dim=1)
            
                # Create decoder time marks
                if Y_time is None:
                    Y_time = torch.zeros((X.size(0), dec_inp.size(1), 5), device=self.device).long()
            
                # Ensure Y_time has the right sequence length
                if Y_time.size(1) < dec_inp.size(1):
                    padding = torch.zeros((Y_time.size(0), dec_inp.size(1) - Y_time.size(1), Y_time.size(2)), device=self.device).long()
                    Y_time_padded = torch.cat([Y_time, padding], dim=1)
                    Y_time = Y_time_padded
                else:
                    Y_time = Y_time[:, :dec_inp.size(1), :]
            
                # Forward pass with Informer model
                predictions = self.model(X, X_time, dec_inp, Y_time)
            else:
                # Standard forward pass for other models
                predictions = self.model(X)  # shape: [B, n_ahead, channels]
        
            loss = self.criterion(predictions, Y)
            if loss.dim() > 0:
                loss = (loss.mean(dim=(1, 2)) * weight).mean()
            loss.backward()
        
            # Apply gradient clipping if configured
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            
            self.optimizer.step()

            total_loss += loss.item() * X.size(0)
            count += X.size(0)
        avg_loss = total_loss / count
        self.metrics["train_loss"].append(avg_loss)
    
        return avg_loss

    def val_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0
        count = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation", leave=False):
                # Handle different batch structures
                if len(batch) == 6:  # Full batch with time data (X, Y, X_time, Y_time, action, weight)
                    X, Y, X_time, Y_time, _, weight = batch
                    has_time_data = True
                elif len(batch) >= 4 and isinstance(batch[2], torch.Tensor) and isinstance(batch[3], torch.Tensor):
                    X, Y, X_time, Y_time = batch[0], batch[1], batch[2], batch[3]
                    actions = batch[4] if len(batch) > 4 and isinstance(batch[4], list) else ["unknown"] * X.size(0)
                    has_time_data = True
                else:
                    # Handle different batch structures without time data
                    X, Y = batch[0], batch[1]
                    X_time, Y_time = None, None
                    has_time_data = False
                    
                    # Get actions from batch if available
                    actions = None
                    if len(batch) > 2:
                        if isinstance(batch[2], list):
                            actions = batch[2]
                        elif len(batch) > 3 and isinstance(batch[3], list):
                            actions = batch[3]
                    
                    if actions is None:
                        actions = ["unknown"] * X.size(0)
            
                X = X.to(self.device)
                Y = Y.to(self.device)
            
                if X_time is not None:
                    X_time = X_time.to(self.device)
                if Y_time is not None:
                    Y_time = Y_time.to(self.device)
            
                # Handle different model types
                if self.model_type == "informer":
                    # Create dummy time features if needed
                    if X_time is None:
                        X_time = torch.zeros((X.size(0), X.size(1), 5), device=self.device).long()
                
                    # Configure decoder input
                    label_len = self.lag // 2
                    dec_inp = torch.zeros((X.size(0), self.n_ahead, X.size(2)), device=self.device)
                    if label_len > 0:
                        dec_inp = torch.cat([Y[:, :label_len, :], dec_inp], dim=1)
                
                    # Create decoder time marks
                    if Y_time is None:
                        Y_time = torch.zeros((X.size(0), dec_inp.size(1), 5), device=self.device).long()
                
                    # Ensure Y_time has the right sequence length
                    if Y_time.size(1) < dec_inp.size(1):
                        padding = torch.zeros((Y_time.size(0), dec_inp.size(1) - Y_time.size(1), Y_time.size(2)), device=self.device).long()
                        Y_time_padded = torch.cat([Y_time, padding], dim=1)
                        Y_time = Y_time_padded
                    else:
                        Y_time = Y_time[:, :dec_inp.size(1), :]
                
                    # Forward pass with Informer model
                    predictions = self.model(X, X_time, dec_inp, Y_time)
                else:
                    # Standard forward pass for other models
                    predictions = self.model(X)
            
            loss = self.criterion(predictions, Y)
            # Compute mean loss over forecast dimensions without weighting
            if loss.dim() > 0:
                loss = loss.mean(dim=(1, 2)).mean()
            total_loss += loss.item() * X.size(0)
            count += X.size(0)
        avg_loss = total_loss / count
        self.metrics["val_loss"].append(avg_loss)
    
        return avg_loss

    def Test_Model(self, loader, save_path=None):
        self.model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in loader:
                # Handle different batch structures
                if len(batch) >= 2:  # All batch structures have at least X and Y
                    X, Y = batch[0], batch[1]
                
                    # Check for time data
                    X_time, Y_time = None, None
                    if len(batch) >= 4 and isinstance(batch[2], torch.Tensor) and isinstance(batch[3], torch.Tensor):
                        X_time, Y_time = batch[2], batch[3]
                else:
                    print("Warning: Unexpected batch structure with less than 2 elements")
                    continue
                
                X = X.to(self.device)
                Y = Y.to(self.device)
            
                if X_time is not None:
                    X_time = X_time.to(self.device)
                if Y_time is not None:
                    Y_time = Y_time.to(self.device)
            
                # Handle different model types
                if self.model_type == "informer":
                    # Create dummy time features if needed
                    if X_time is None:
                        X_time = torch.zeros((X.size(0), X.size(1), 5), device=self.device).long()
                
                    # Configure decoder input
                    label_len = self.lag // 2
                    dec_inp = torch.zeros((X.size(0), self.n_ahead, X.size(2)), device=self.device)
                    if label_len > 0:
                        dec_inp = torch.cat([Y[:, :label_len, :], dec_inp], dim=1)
                
                    # Create decoder time marks
                    if Y_time is None:
                        Y_time = torch.zeros((X.size(0), dec_inp.size(1), 5), device=self.device).long()
                
                    # Ensure Y_time has the right sequence length
                    if Y_time.size(1) < dec_inp.size(1):
                        padding = torch.zeros((Y_time.size(0), dec_inp.size(1) - Y_time.size(1), Y_time.size(2)), device=self.device).long()
                        Y_time_padded = torch.cat([Y_time, padding], dim=1)
                        Y_time = Y_time_padded
                    else:
                        Y_time = Y_time[:, :dec_inp.size(1), :]
                
                    # Forward pass with Informer model
                    preds = self.model(X, X_time, dec_inp, Y_time)
                else:
                    # Standard forward pass for other models
                    preds = self.model(X)
                
                all_preds.append(preds)
                all_targets.append(Y)
    
        self.test_results = {
            'preds': torch.cat(all_preds, dim=0),
            'targets': torch.cat(all_targets, dim=0)
        }
    
        if save_path is not None:
            self.plot_test_results(test_loader=loader, save_dir=save_path)
    
        return self.test_results

    def step_scheduler(self, metrics=None):
        if self.scheduler is not None:
            # Check if scheduler is ReduceLROnPlateau which requires metrics
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if metrics is None and len(self.metrics["val_loss"]) > 0:
                    # Use the latest validation loss if metrics not provided
                    metrics = self.metrics["val_loss"][-1]
                self.scheduler.step(metrics)
            else:
                # Other schedulers like StepLR don't need metrics
                self.scheduler.step()

    def save_first_windows(self, train_loader, epoch, save_dir, num_windows=10):
        """
        Save a given number of sample forecast windows from the training set.
        """
        self.model.eval()
        os.makedirs(save_dir, exist_ok=True)
        
        window_count = 0
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Saving Windows", leave=False):
                # Handle different batch structures
                if len(batch) == 6:  # Full batch with time data (X, Y, X_time, Y_time, action, weight)
                    X_batch, Y_batch, X_time, Y_time, actions, _ = batch
                    has_time_data = True
                elif len(batch) >= 4 and isinstance(batch[2], torch.Tensor) and isinstance(batch[3], torch.Tensor):
                    X_batch, Y_batch, X_time, Y_time = batch[0], batch[1], batch[2], batch[3]
                    actions = batch[4] if len(batch) > 4 and isinstance(batch[4], list) else ["unknown"] * X_batch.size(0)
                    has_time_data = True
                else:
                    # Handle different batch structures without time data
                    X_batch, Y_batch = batch[0], batch[1]
                    X_time, Y_time = None, None
                    has_time_data = False
                    
                    # Get actions from batch if available
                    actions = None
                    if len(batch) > 2:
                        if isinstance(batch[2], list):
                            actions = batch[2]
                        elif len(batch) > 3 and isinstance(batch[3], list):
                            actions = batch[3]
                    
                    if actions is None:
                        actions = ["unknown"] * X_batch.size(0)
            
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)
            
                if X_time is not None:
                    X_time = X_time.to(self.device)
                if Y_time is not None:
                    Y_time = Y_time.to(self.device)
            
                batch_size = X_batch.size(0)
                for i in range(batch_size):
                    if window_count >= num_windows:
                        break
                        
                    # Get ground truth
                    gt_forecast = Y_batch[i].detach().cpu().numpy()  # expected: [n_ahead, channels]
                    
                    # Get prediction based on model type
                    if self.model_type == "informer":
                        # Prepare inputs for Informer model
                        x_enc = X_batch[i].unsqueeze(0)  # Add batch dimension
                        
                        # Create dummy time features if needed
                        if X_time is None:
                            x_mark_enc = torch.zeros((1, x_enc.size(1), 5), device=self.device).long()
                        else:
                            x_mark_enc = X_time[i].unsqueeze(0)
                    
                        # Configure decoder input
                        label_len = self.lag // 2
                        dec_inp = torch.zeros((1, self.n_ahead, x_enc.size(2)), device=self.device)
                        if label_len > 0:
                            dec_inp = torch.cat([Y_batch[i, :label_len, :].unsqueeze(0), dec_inp], dim=1)
                    
                        # Create decoder time marks
                        if Y_time is None:
                            x_mark_dec = torch.zeros((1, dec_inp.size(1), 5), device=self.device).long()
                        else:
                            # Ensure Y_time has the right sequence length
                            if Y_time[i].size(0) < dec_inp.size(1):
                                padding = torch.zeros((dec_inp.size(1) - Y_time[i].size(0), Y_time.size(2)), device=self.device).long()
                                y_time_padded = torch.cat([Y_time[i], padding], dim=0)
                                x_mark_dec = y_time_padded.unsqueeze(0)
                            else:
                                x_mark_dec = Y_time[i, :dec_inp.size(1), :].unsqueeze(0)
                    
                        # Get prediction from Informer model
                        pred_forecast = self.model(x_enc, x_mark_enc, dec_inp, x_mark_dec)
                    else:
                        # Standard forward pass for other models
                        pred_forecast = self.model(X_batch[i].unsqueeze(0))
                
                    if isinstance(pred_forecast, tuple):  # Handle models that return multiple outputs
                        pred_forecast = pred_forecast[0]
                        
                    pred_forecast = pred_forecast.squeeze(0).detach().cpu().numpy()
                    
                    # Save the plot
                    save_path = os.path.join(save_dir, f"window_{window_count+1}_epoch_{epoch}.png")
                    
                    # Get action if available
                    action = actions[i] if isinstance(actions, list) and i < len(actions) else None
                    
                    self.plot_prediction(
                        X_batch[i].detach().cpu().numpy(),
                        gt_forecast,
                        pred_forecast,
                        save_path,
                        action=action
                    )
                    
                    window_count += 1
                    
                if window_count >= num_windows:
                    break
                    
        print(f"Saved {window_count} training windows to {save_dir}")

    def plot_validation_results(self, val_loader, save_dir, num_windows=3):
        """
        Plot validation results for visual inspection
        """
        self.model.eval()
        os.makedirs(save_dir, exist_ok=True)
        
        # Keep track of how many samples we've plotted for each action
        action_counts = {}
        
        with torch.no_grad():
            for batch in val_loader:
                # Handle different batch structures
                if len(batch) == 6:  # Full batch with time data (X, Y, X_time, Y_time, action, weight)
                    X, Y, X_time, Y_time, actions, _ = batch
                    has_time_data = True
                elif len(batch) >= 4 and isinstance(batch[2], torch.Tensor) and isinstance(batch[3], torch.Tensor):
                    X, Y, X_time, Y_time = batch[0], batch[1], batch[2], batch[3]
                    actions = batch[4] if len(batch) > 4 and isinstance(batch[4], list) else ["unknown"] * X.size(0)
                    has_time_data = True
                else:
                    # Handle different batch structures without time data
                    X, Y = batch[0], batch[1]
                    X_time, Y_time = None, None
                    has_time_data = False
                    
                    # Get actions from batch if available
                    actions = None
                    if len(batch) > 2:
                        if isinstance(batch[2], list):
                            actions = batch[2]
                        elif len(batch) > 3 and isinstance(batch[3], list):
                            actions = batch[3]
                    
                    if actions is None:
                        actions = ["unknown"] * X.size(0)
            
                X = X.to(self.device)
                Y = Y.to(self.device)
            
                if X_time is not None:
                    X_time = X_time.to(self.device)
                if Y_time is not None:
                    Y_time = Y_time.to(self.device)
            
                # Process each sample in the batch
                for i in range(X.size(0)):
                    action = actions[i] if isinstance(actions, list) and i < len(actions) else "unknown"
                    
                    # Skip if we already have enough samples for this action
                    if action_counts.get(action, 0) >= num_windows:
                        continue
                    
                    # Create action-specific directory
                    action_dir = os.path.join(save_dir, f"action_{action}")
                    os.makedirs(action_dir, exist_ok=True)
                    
                    # Get prediction based on model type
                    if self.model_type == "informer":
                        # Prepare inputs for Informer model
                        x_enc = X[i].unsqueeze(0)  # Add batch dimension
                        
                        # Create dummy time features if needed
                        if X_time is None:
                            x_mark_enc = torch.zeros((1, x_enc.size(1), 5), device=self.device).long()
                        else:
                            x_mark_enc = X_time[i].unsqueeze(0)
                    
                        # Configure decoder input
                        label_len = self.lag // 2
                        dec_inp = torch.zeros((1, self.n_ahead, x_enc.size(2)), device=self.device)
                        if label_len > 0:
                            dec_inp = torch.cat([Y[i, :label_len, :].unsqueeze(0), dec_inp], dim=1)
                    
                        # Create decoder time marks
                        if Y_time is None:
                            x_mark_dec = torch.zeros((1, dec_inp.size(1), 5), device=self.device).long()
                        else:
                            # Ensure Y_time has the right sequence length
                            if Y_time[i].size(0) < dec_inp.size(1):
                                padding = torch.zeros((dec_inp.size(1) - Y_time[i].size(0), Y_time.size(2)), device=self.device).long()
                                y_time_padded = torch.cat([Y_time[i], padding], dim=0)
                                x_mark_dec = y_time_padded.unsqueeze(0)
                            else:
                                x_mark_dec = Y_time[i, :dec_inp.size(1), :].unsqueeze(0)
                    
                        # Get prediction from Informer model
                        prediction = self.model(x_enc, x_mark_enc, dec_inp, x_mark_dec).squeeze(0)
                    else:
                        # Standard forward pass for other models
                        prediction = self.model(X[i].unsqueeze(0)).squeeze(0)
                
                    # Plot and save
                    save_path = os.path.join(action_dir, f"val_sample_{action_counts.get(action, 0)+1}.png")
                    self.plot_prediction(
                        X[i].detach().cpu().numpy(),
                        Y[i].detach().cpu().numpy(),
                        prediction.detach().cpu().numpy(),
                        save_path,
                        action=action
                    )
                    
                    action_counts[action] = action_counts.get(action, 0) + 1

    def plot_test_results(self, test_loader, save_dir, num_windows=3):
        """
        Computes the average test loss over the entire test set and
        plots a given number of sample forecast windows from the test set,
        displaying the average test loss in the title of each plot.
        """
        self.model.eval()
        os.makedirs(save_dir, exist_ok=True)
        
        # Keep track of how many samples we've plotted for each action
        action_counts = {}
        
        # Calculate average test loss
        total_loss = 0
        count = 0
        with torch.no_grad():
            for batch in test_loader:
                # Handle different batch structures
                if len(batch) == 6:  # Full batch with time data (X, Y, X_time, Y_time, action, weight)
                    X_batch, Y_batch, X_time, Y_time, _, _ = batch
                    has_time_data = True
                elif len(batch) >= 4 and isinstance(batch[2], torch.Tensor) and isinstance(batch[3], torch.Tensor):
                    X_batch, Y_batch, X_time, Y_time = batch[0], batch[1], batch[2], batch[3]
                    has_time_data = True
                else:
                    # Handle different batch structures without time data
                    X_batch, Y_batch = batch[0], batch[1]
                    X_time, Y_time = None, None
                    has_time_data = False
                
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)
            
                if X_time is not None:
                    X_time = X_time.to(self.device)
                if Y_time is not None:
                    Y_time = Y_time.to(self.device)
            
                # Get predictions based on model type
                if self.model_type == "informer":
                    # Create dummy time features if needed
                    if X_time is None:
                        X_time = torch.zeros((X_batch.size(0), X_batch.size(1), 5), device=self.device).long()
                
                    # Configure decoder input
                    label_len = self.lag // 2
                    dec_inp = torch.zeros((X_batch.size(0), self.n_ahead, X_batch.size(2)), device=self.device)
                    if label_len > 0:
                        dec_inp = torch.cat([Y_batch[:, :label_len, :], dec_inp], dim=1)
                
                    # Create decoder time marks
                    if Y_time is None:
                        Y_time = torch.zeros((X_batch.size(0), dec_inp.size(1), 5), device=self.device).long()
                
                    # Ensure Y_time has the right sequence length
                    if Y_time.size(1) < dec_inp.size(1):
                        padding = torch.zeros((Y_time.size(0), dec_inp.size(1) - Y_time.size(1), Y_time.size(2)), device=self.device).long()
                        Y_time_padded = torch.cat([Y_time, padding], dim=1)
                        Y_time = Y_time_padded
                    else:
                        Y_time = Y_time[:, :dec_inp.size(1), :]
                
                    # Forward pass with Informer model
                    preds = self.model(X_batch, X_time, dec_inp, Y_time)
                else:
                    # Standard forward pass for other models
                    preds = self.model(X_batch)
                
                loss = self.criterion(preds, Y_batch)
                if loss.dim() > 0:
                    loss = (loss.mean(dim=(1,2))).mean()
                total_loss += loss.item() * X_batch.size(0)
                count += X_batch.size(0)
    
        avg_test_loss = total_loss / count
        print(f"Average Test Loss: {avg_test_loss:.6f}")
    
        # Plot sample windows
        with torch.no_grad():
            for batch in test_loader:
                # Handle different batch structures
                if len(batch) == 6:  # Full batch with time data (X, Y, X_time, Y_time, action, weight)
                    X, Y, X_time, Y_time, actions, _ = batch
                    has_time_data = True
                elif len(batch) >= 4 and isinstance(batch[2], torch.Tensor) and isinstance(batch[3], torch.Tensor):
                    X, Y, X_time, Y_time = batch[0], batch[1], batch[2], batch[3]
                    actions = batch[4] if len(batch) > 4 and isinstance(batch[4], list) else ["unknown"] * X.size(0)
                    has_time_data = True
                else:
                    # Handle different batch structures without time data
                    X, Y = batch[0], batch[1]
                    X_time, Y_time = None, None
                    has_time_data = False
                    
                    # Get actions from batch if available
                    actions = None
                    if len(batch) > 2:
                        if isinstance(batch[2], list):
                            actions = batch[2]
                        elif len(batch) > 3 and isinstance(batch[3], list):
                            actions = batch[3]
                    
                    if actions is None:
                        actions = ["unknown"] * X.size(0)
            
                X = X.to(self.device)
                Y = Y.to(self.device)
            
                if X_time is not None:
                    X_time = X_time.to(self.device)
                if Y_time is not None:
                    Y_time = Y_time.to(self.device)
            
                # Process each sample in the batch
                for i in range(X.size(0)):
                    action = actions[i] if isinstance(actions, list) and i < len(actions) else "unknown"
                    
                    # Skip if we already have enough samples for this action
                    if action_counts.get(action, 0) >= num_windows:
                        continue
                    
                    # Create action-specific directory
                    action_dir = os.path.join(save_dir, f"action_{action}")
                    os.makedirs(action_dir, exist_ok=True)
                    
                    # Get prediction based on model type
                    if self.model_type == "informer":
                        # Prepare inputs for Informer model
                        x_enc = X[i].unsqueeze(0)  # Add batch dimension
                        
                        # Create dummy time features if needed
                        if X_time is None:
                            x_mark_enc = torch.zeros((1, x_enc.size(1), 5), device=self.device).long()
                        else:
                            x_mark_enc = X_time[i].unsqueeze(0)
                    
                        # Configure decoder input
                        label_len = self.lag // 2
                        dec_inp = torch.zeros((1, self.n_ahead, x_enc.size(2)), device=self.device)
                        if label_len > 0:
                            dec_inp = torch.cat([Y[i, :label_len, :].unsqueeze(0), dec_inp], dim=1)
                    
                        # Create decoder time marks
                        if Y_time is None:
                            x_mark_dec = torch.zeros((1, dec_inp.size(1), 5), device=self.device).long()
                        else:
                            # Ensure Y_time has the right sequence length
                            if Y_time[i].size(0) < dec_inp.size(1):
                                padding = torch.zeros((dec_inp.size(1) - Y_time[i].size(0), Y_time.size(2)), device=self.device).long()
                                y_time_padded = torch.cat([Y_time[i], padding], dim=0)
                                x_mark_dec = y_time_padded.unsqueeze(0)
                            else:
                                x_mark_dec = Y_time[i, :dec_inp.size(1), :].unsqueeze(0)
                    
                        # Get prediction from Informer model
                        prediction = self.model(x_enc, x_mark_enc, dec_inp, x_mark_dec).squeeze(0)
                    else:
                        # Standard forward pass for other models
                        prediction = self.model(X[i].unsqueeze(0)).squeeze(0)
                
                    # Plot and save
                    save_path = os.path.join(action_dir, f"test_sample_{action_counts.get(action, 0)+1}_loss_{avg_test_loss:.4f}.png")
                    
                    self.plot_prediction(
                        X[i].detach().cpu().numpy(),
                        Y[i].detach().cpu().numpy(),
                        prediction.detach().cpu().numpy(),
                        save_path,
                        action=action
                    )
                    
                    action_counts[action] = action_counts.get(action, 0) + 1
                    
                    # Break if we have enough samples for all actions
                    if all(count >= num_windows for count in action_counts.values()) and len(action_counts) > 0:
                        break
                
                # Break if we have enough samples for all actions
                if all(count >= num_windows for count in action_counts.values()) and len(action_counts) > 0:
                    break
    
        print(f"Test results saved to {save_dir}")
        return avg_test_loss

    def plot_prediction(self, X_sample, Y_sample, Y_pred, fig_path, action=None):
        # Ensure all tensors are on CPU and converted to numpy
        if isinstance(X_sample, torch.Tensor):
            X_sample = X_sample.detach().cpu().numpy()
        if isinstance(Y_sample, torch.Tensor):
            Y_sample = Y_sample.detach().cpu().numpy()
        if isinstance(Y_pred, torch.Tensor):
            Y_pred = Y_pred.detach().cpu().numpy()
        
        # Make sure we have at least 2D arrays for processing
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(-1, 1)
        
        # Handle different dimensions for Y_sample and Y_pred
        # If we have a batch dimension, take the first sample
        if Y_sample.ndim == 3:  # [batch, n_ahead, channels]
            Y_sample = Y_sample[0]  # Take first sample if batched
        if Y_sample.ndim == 1:
            Y_sample = Y_sample.reshape(-1, 1)
        
        if Y_pred.ndim == 3:  # [batch, n_ahead, channels]
            Y_pred = Y_pred[0]  # Take first sample if batched
        if Y_pred.ndim == 1:
            Y_pred = Y_pred.reshape(-1, 1)
        
        # Handle different output shapes from models
        # If the first dimension is batch size (1), squeeze it out
        if Y_pred.shape[0] == 1 and Y_pred.ndim > 2:  # [1, n_ahead, channels]
            Y_pred = Y_pred.squeeze(0)
        
        # Make sure forecasts have [n_ahead, channels] shape
        if Y_sample.shape[0] != self.n_ahead:
            # If forecast shape is [channels, n_ahead]
            if Y_sample.shape[1] == self.n_ahead:
                Y_sample = Y_sample.T
            # If forecast is a flattened vector, reshape it
            elif Y_sample.size == self.n_ahead:
                Y_sample = Y_sample.reshape(self.n_ahead, -1)

        if Y_pred.shape[0] != self.n_ahead:
            # If prediction shape is [channels, n_ahead]
            if Y_pred.shape[1] == self.n_ahead:
                Y_pred = Y_pred.T
            # If prediction is a flattened vector, reshape it
            elif Y_pred.size == self.n_ahead:
                Y_pred = Y_pred.reshape(self.n_ahead, -1)

        n_lag = X_sample.shape[0]
        n_channels_lag = X_sample.shape[1]
        n_channels_forecast = Y_sample.shape[1] if Y_sample.ndim > 1 else 1
        n_channels_pred = Y_pred.shape[1] if Y_pred.ndim > 1 else 1

        # Use the minimum number of channels between ground truth and prediction
        n_channels_to_plot = min(n_channels_forecast, n_channels_pred)

        # Generate descriptive channel names based on sensor_mode
        if self.sensor_mode == "emg":
            # Create names like "EMG 1", "EMG 2", etc.
            lag_labels = [f"EMG {i+1}" for i in range(n_channels_lag)]
            forecast_labels = [f"EMG {i+1}" for i in range(n_channels_forecast)]
            input_marker = 'o'  # Circle for EMG inputs
            output_marker = 'o'  # Circle for EMG outputs
        elif self.sensor_mode == "acc":
            # Create names for accelerometer channels
            lag_labels = []
            forecast_labels = []
            for i in range(n_channels_lag // 3):
                for axis in ["X", "Y", "Z"]:
                    lag_labels.append(f"ACC {i+1} {axis}")
            for i in range(n_channels_forecast // 3):
                for axis in ["X", "Y", "Z"]:
                    forecast_labels.append(f"ACC {i+1} {axis}")
            # Add any remaining channels
            for i in range(len(lag_labels), n_channels_lag):
                lag_labels.append(f"ACC {i+1}")
            for i in range(len(forecast_labels), n_channels_forecast):
                forecast_labels.append(f"ACC {i+1}")
            input_marker = 's'  # Square for ACC inputs
            output_marker = 's'  # Square for ACC outputs
        elif self.sensor_mode == "gyro":
            # Create names for gyroscope channels
            lag_labels = []
            forecast_labels = []
            for i in range(n_channels_lag // 3):
                for axis in ["X", "Y", "Z"]:
                    lag_labels.append(f"GYRO {i+1} {axis}")
            for i in range(n_channels_forecast // 3):
                for axis in ["X", "Y", "Z"]:
                    forecast_labels.append(f"GYRO {i+1} {axis}")
            # Add any remaining channels
            for i in range(len(lag_labels), n_channels_lag):
                lag_labels.append(f"GYRO {i+1}")
            for i in range(len(forecast_labels), n_channels_forecast):
                forecast_labels.append(f"GYRO {i+1}")
            input_marker = '^'  # Triangle for GYRO inputs
            output_marker = '^'  # Triangle for GYRO outputs
        elif self.sensor_mode == "all":
            # For mixed sensor types, use a generic naming scheme
            lag_labels = [f"Channel {i+1}" for i in range(n_channels_lag)]
            forecast_labels = [f"Channel {i+1}" for i in range(n_channels_forecast)]
            input_marker = 'o'  # Circle for generic inputs
            output_marker = 'o'  # Circle for generic outputs
        else:
            # Generic fallback
            lag_labels = [f"{self.sensor_mode.upper()} {i+1}" for i in range(n_channels_lag)]
            forecast_labels = [f"{self.sensor_mode.upper()} {i+1}" for i in range(n_channels_forecast)]
            input_marker = 'o'  # Circle for generic inputs
            output_marker = 'o'  # Circle for generic outputs

        # Define color palettes for inputs and outputs
        INPUT_COLORS = [
            '#800000', '#008000', '#000080',  # Darker shades
            '#804000', '#408000', '#004080',
            '#400080', '#800040', '#008040',
            '#404080', '#804040', '#408040'
        ]
        OUTPUT_COLORS = [
            '#ff0000', '#00ff00', '#0000ff',  # Brighter shades
            '#ff8000', '#80ff00', '#0080ff',
            '#8000ff', '#ff0080', '#00ff80',
            '#8080ff', '#ff8080', '#80ff80'
        ]

        # Use sequential timesteps - ensure x_forecast has the right length
        x_lag = np.arange(0, n_lag)
        x_forecast = np.arange(n_lag, n_lag + Y_sample.shape[0])

        # Create figure with subplots for each channel
        fig, axes = plt.subplots(n_channels_to_plot, 1, figsize=(12, 2 * n_channels_to_plot), sharex=True)
        
        # If we only have one channel, wrap it in a list to handle indexing
        if n_channels_to_plot == 1:
            axes = [axes]

        # Plot each channel in its own subplot
        for ch in range(n_channels_to_plot):
            ax = axes[ch]
            
            # Get the appropriate color for this channel
            channel_color = get_channel_color(ch)
            # Find the matching index in OUTPUT_COLORS and use the corresponding INPUT_COLORS
            output_color_index = OUTPUT_COLORS.index(channel_color) if channel_color in OUTPUT_COLORS else ch % len(INPUT_COLORS)
            input_color = INPUT_COLORS[output_color_index]
            
            # Plot input sequence
            ax.plot(x_lag, X_sample[:, ch], 
                    color=input_color, 
                    marker=input_marker,
                    linestyle='-',
                    linewidth=1.5,
                    alpha=0.7,
                    label='Input')
            
            # Plot ground truth
            ax.plot(x_forecast, Y_sample[:, ch], 
                    color=channel_color,
                    marker=output_marker,
                    linestyle=GROUND_TRUTH_LINESTYLE,
                    linewidth=2.0,
                    alpha=0.9,
                    label='Ground Truth')
            
            # Plot prediction
            if Y_pred.shape[0] != len(x_forecast):
                if Y_pred.shape[0] > len(x_forecast):
                    Y_pred_plot = Y_pred[:len(x_forecast), ch]
                else:
                    padding = np.full(len(x_forecast) - Y_pred.shape[0], Y_pred[-1, ch])
                    Y_pred_plot = np.concatenate([Y_pred[:, ch], padding])
            else:
                Y_pred_plot = Y_pred[:, ch]
            
            ax.plot(x_forecast, Y_pred_plot,
                    color=channel_color,
                    marker=output_marker,
                    linestyle=PREDICTION_LINESTYLE,
                    linewidth=2.0,
                    alpha=0.9,
                    label='Prediction')
            
            # Add vertical line at transition
            ax.axvline(x=n_lag, color='gray', linestyle=':', linewidth=1)
            
            # Set title and labels
            ax.set_title(f"{forecast_labels[ch]}")
            ax.set_ylabel("Signal Value")
            
            # Add legend
            ax.legend(loc='upper right', fontsize=8)
            
            # Grid and limits
            ax.grid(True)
            ax.set_xlim(0, n_lag + Y_sample.shape[0])
            
        # Add x-axis label to bottom plot
        axes[-1].set_xlabel("Time Step")
        
        # Add main title
        title = f"{self.sensor_mode.upper()} Channel Predictions"
        if action is not None:
            title += f" - Action {action}"
        fig.suptitle(title, y=1.02)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save and close
        plt.savefig(fig_path, bbox_inches='tight')
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

    def log_epoch_loss(self, epoch, train_loss, val_loss, extra_metrics, log_file_path):
        header = "epoch,train_loss,val_loss,rmse,mae,r2_score,corr_coef\n"
        # If the file does not exist, create it and write a header.
        if not os.path.exists(log_file_path):
            with open(log_file_path, "w") as f:
                f.write(header)
        # Append this epoch’s metrics.
        with open(log_file_path, "a") as f:
            f.write(f"{epoch},{train_loss},{val_loss},"
                    f"{extra_metrics['rmse']},{extra_metrics['mae']},"
                    f"{extra_metrics['r2_score']},{extra_metrics['corr_coef']}\n")

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

            # Compute and log additional validation metrics using the new function.
            metrics = self.compute_metrics(val_loader)
            print(f"Validation RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2_score']:.4f}, Corr: {metrics['corr_coef']:.4f}")

            # Log the per-epoch metrics if a log file has been specified.
            if self.epoch_log_file:
                self.log_epoch_loss(epoch+1, train_loss, val_loss, metrics, self.epoch_log_file)
                
            # Save per-epoch checkpoint
            epoch_checkpoint_path = f"{epoch_checkpoint_prefix}_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, epoch_checkpoint_path)
            print(f"Epoch checkpoint saved to {epoch_checkpoint_path}")
            
            # Save sample forecast windows for this epoch
            self.save_first_windows(train_loader, epoch+1, save_dir=os.path.join(self.fig_dir, self.model_type, f"epoch_{epoch+1}"), num_windows=num_windows)
            
            # Early stopping check (saves best checkpoint automatically)
            early_stopping(val_loss, train_loss, self.model)  # Updated to pass train_loss
            if early_stopping.early_stop:
                print("Early stopping triggered. Stopping training.")
                break
            
            # Step the scheduler if applicable
            self.step_scheduler(val_loss)
        
        # After training, plot and save the loss curve
        self.plot_loss_curve(loss_curve_path)
        
        # Return the metrics dictionary
        return self.metrics

    def compute_metrics(self, dataloader):
        """
        Computes evaluation metrics (e.g., RMSE) for the model on the given dataloader.
        Returns a dictionary of metrics.
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Check if batch has time data (assume at least 4 elements for Informer)
                if self.model_type == "informer" and len(batch) >= 4:
                    X, Y, X_time, Y_time = batch[0], batch[1], batch[2], batch[3]
                    X = X.to(self.device)
                    Y = Y.to(self.device)
                    X_time = X_time.to(self.device)
                    Y_time = Y_time.to(self.device)

                    # Configure decoder input (using label length = lag // 2)
                    label_len = self.lag // 2
                    dec_inp = torch.zeros((X.size(0), self.n_ahead, X.size(2)), device=self.device)
                    if label_len > 0:
                        dec_inp = torch.cat([Y[:, :label_len, :], dec_inp], dim=1)
                    
                    # Ensure Y_time (decoder time marks) has proper dimensions
                    if Y_time.size(1) < dec_inp.size(1):
                        padding = torch.zeros((Y_time.size(0), dec_inp.size(1) - Y_time.size(1), Y_time.size(2)), device=self.device).long()
                        Y_time = torch.cat([Y_time, padding], dim=1)
                    else:
                        Y_time = Y_time[:, :dec_inp.size(1), :]

                    preds = self.model(X, X_time, dec_inp, Y_time)
                else:
                    # For models that don't use time features.
                    X, Y = batch[0], batch[1]
                    X = X.to(self.device)
                    Y = Y.to(self.device)
                    preds = self.model(X)

                all_preds.append(preds.cpu().numpy())
                all_targets.append(Y.cpu().numpy())
        
        # Concatenate all predictions and targets
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Compute metrics
        mse = np.mean((all_preds - all_targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(all_preds - all_targets))
        r2 = r2_score(all_targets.flatten(), all_preds.flatten())
        corr_coef = np.corrcoef(all_targets.flatten(), all_preds.flatten())[0, 1]

        return {"rmse": rmse, "mae": mae, "r2_score": r2, "corr_coef": corr_coef}
