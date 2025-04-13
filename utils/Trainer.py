import os
import re
import glob
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

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
    def __init__(self, patience=10, min_delta=0.0, verbose=False, path='checkpoint.pt', start_epoch=20):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.path = path
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.start_epoch = start_epoch
        self.current_epoch = 0

    def __call__(self, val_loss, model, current_epoch):
        self.current_epoch = current_epoch
        # Only check for early stopping after start_epoch
        if self.current_epoch < self.start_epoch:
            return False
        
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            if self.verbose:
                print(f"Initial validation loss: {val_loss:.8f}. Saving model.")
        elif val_loss < self.best_loss - self.min_delta:
            improvement = self.best_loss - val_loss
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            if self.verbose:
                print(f"Validation loss decreased by {improvement:.8f} to {val_loss:.8f}. Saving best model and resetting counter.")
        else:
            self.counter += 1
            if self.verbose:
                diff = val_loss - self.best_loss
                print(f"Validation loss did not improve. Current: {val_loss:.8f}, Best: {self.best_loss:.8f}, Diff: {diff:.8f}")
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered. Stopping training.")
        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Saving best model checkpoint with validation loss: {val_loss}")
        torch.save(model.state_dict(), self.path)

class Trainer:
    def __init__(self, model, lag, n_ahead, optimizer, scheduler, testloader, fig_dir,
                 loss='custom', model_type="lstm", device="cpu", is_classification=False, clip_grad_norm=1.0, **kwargs):
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
        self.debug = kwargs.get('debug', False)  # Add debug flag
        
        # Check if model_type is informer to handle time data
        self.is_informer = self.model_type == "informer"
        
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
    
    def _print_gradient_norms(self):
        """Print gradient norms for each parameter in the model."""
        total_norm = 0.0
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                print(f"Gradient norm for {name}: {param_norm.item():.4f}")
        total_norm = total_norm ** 0.5
        print(f"Total gradient norm: {total_norm:.4f}")

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Configure progress bar
        t = tqdm(dataloader, desc="Training", leave=False)
        
        for batch in t:
            if self.is_informer and len(batch) > 4:  # Check if time data is included in the batch
                # Unpack batch with time data
                if len(batch) == 6:  # X, Y, X_time, Y_time, action, weight
                    X, Y, X_time, Y_time, actions, weights = batch
                else:  # Fallback if batch structure is different
                    X, Y = batch[0], batch[1]
                    X_time, Y_time = batch[2], batch[3] if len(batch) > 3 else (None, None)
                    weights = batch[-1] if isinstance(batch[-1], torch.Tensor) else torch.ones_like(Y)
                
                X = X.to(self.device)
                Y = Y.to(self.device)
                X_time = X_time.to(self.device) if X_time is not None else None
                Y_time = Y_time.to(self.device) if Y_time is not None else None
                weights = weights.to(self.device)
                
                # Forward pass for Informer model
                self.optimizer.zero_grad()
                
                # Configure decoder input (use the standard approach for Informer)
                label_len = self.lag // 2  # Half of lag is used as label portion
                dec_inp = torch.zeros((X.size(0), self.n_ahead, X.size(2)), device=self.device)
                dec_inp = torch.cat([Y[:, :label_len, :], dec_inp], dim=1) if label_len > 0 else dec_inp
                
                # Make sure both encoder and decoder time marks have the right sequence lengths
                # The encoder time marks should have the same sequence length as X
                # The decoder time marks should have the same sequence length as dec_inp
                
                # 1. Prepare encoder time marks (X_time)
                if X_time is not None:
                    # Ensure X_time has the same sequence length as X
                    encoder_seq_len = X.size(1)
                    if X_time.size(1) != encoder_seq_len:
                        if X_time.size(1) < encoder_seq_len:
                            # Pad X_time to match X length
                            padding_len = encoder_seq_len - X_time.size(1)
                            last_time_entries = X_time[:, -1:, :].repeat(1, padding_len, 1)
                            enc_time_mark = torch.cat([X_time, last_time_entries], dim=1)
                        else:
                            # Slice X_time to match X length
                            enc_time_mark = X_time[:, :encoder_seq_len, :]
                    else:
                        enc_time_mark = X_time
                else:
                    # Create dummy encoder time marks if none provided
                    enc_time_mark = torch.zeros((X.size(0), X.size(1), 5), device=self.device).long()
                
                # 2. Prepare decoder time marks
                decoder_seq_len = dec_inp.size(1)  # This is label_len + pred_len
                if Y_time is not None:
                    # Create a new time tensor with the correct sequence length
                    if Y_time.size(1) < decoder_seq_len:
                        # Pad Y_time to match dec_inp length
                        padding_len = decoder_seq_len - Y_time.size(1)
                        last_time_entries = Y_time[:, -1:, :].repeat(1, padding_len, 1)
                        dec_time_mark = torch.cat([Y_time, last_time_entries], dim=1)
                    else:
                        # Slice Y_time to match dec_inp length
                        dec_time_mark = Y_time[:, :decoder_seq_len, :]
                else:
                    # Create dummy decoder time marks if none provided
                    dec_time_mark = torch.zeros((X.size(0), decoder_seq_len, 5), device=self.device).long()
                
                # Use X, enc_time_mark, dec_inp, dec_time_mark as inputs to the Informer model
                predictions = self.model(X, enc_time_mark, dec_inp, dec_time_mark)
            else:
                X, Y = batch[0], batch[1]
                X = X.to(self.device)
                Y = Y.to(self.device)
                weights = batch[-1].to(self.device) if isinstance(batch[-1], torch.Tensor) else torch.ones_like(Y)
                
                # Forward pass - standard approach
                self.optimizer.zero_grad()
                predictions = self.model(X)
            
            if self.criterion == custom_loss:
                loss = self.criterion(predictions, Y)
            else:
                loss = self.criterion(predictions, Y)
                # Fix the dimension mismatch in weights application
                # First check tensor shapes
                if weights.dim() == 1:
                    # Expand weights to match loss dimensions properly
                    if loss.dim() > 1:
                        # Reshape weights to match the loss dimensions
                        # This ensures proper broadcasting
                        for i in range(1, loss.dim()):
                            weights = weights.unsqueeze(i)
                    loss = (loss * weights).mean()
                else:
                    # If weights already have multiple dimensions, ensure they match loss dimensions
                    if weights.shape != loss.shape:
                        # Reshape or broadcast weights to match loss shape
                        weights = weights.view(weights.size(0), -1)
                        if weights.size(1) == 1:
                            # Broadcast single weight per sample to all elements
                            weights = weights.expand(-1, loss.size(1) if loss.dim() > 1 else 1)
                    loss = (loss * weights).mean()
            
            # Backward pass and optimize
            loss.backward()
            
            # Debug: Print gradient norms periodically
            if hasattr(self, 'debug') and self.debug and num_batches % 10 == 0:
                self._print_gradient_norms()
            
            # Gradient clipping
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                
            self.optimizer.step()
            
            # Update running loss
            try:
                # Move loss to CPU first to avoid CUDA misaligned memory issues
                loss_value = loss.detach().cpu().item()
                total_loss += loss_value
            except (RuntimeError, ValueError) as e:
                print(f"Warning: Could not retrieve loss value during training: {e}")
                # Use a small constant instead to avoid breaking the loop
                total_loss += 0.0
                
            num_batches += 1
            
            # Update progress bar
            t.set_postfix({"batch_loss": loss_value if 'loss_value' in locals() else 0.0})
            
        avg_loss = total_loss / max(1, num_batches)
        self.metrics["train_loss"].append(avg_loss)
        return avg_loss


    def val_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation", leave=False):
                if self.is_informer and len(batch) > 4:  # Check if time data is included in the batch
                    # Unpack batch with time data
                    if len(batch) == 6:  # X, Y, X_time, Y_time, action, weight
                        X, Y, X_time, Y_time, actions, weights = batch
                    else:  # Fallback if batch structure is different
                        X, Y = batch[0], batch[1]
                        X_time, Y_time = batch[2], batch[3] if len(batch) > 3 else (None, None)
                        weights = batch[-1] if isinstance(batch[-1], torch.Tensor) else torch.ones_like(Y)
                    
                    X = X.to(self.device)
                    Y = Y.to(self.device)
                    X_time = X_time.to(self.device) if X_time is not None else None
                    Y_time = Y_time.to(self.device) if Y_time is not None else None
                    weights = weights.to(self.device)
                    
                    # Configure decoder input for Informer
                    label_len = self.lag // 2  # Half of lag is used as label portion
                    dec_inp = torch.zeros((X.size(0), self.n_ahead, X.size(2)), device=self.device)
                    dec_inp = torch.cat([Y[:, :label_len, :], dec_inp], dim=1) if label_len > 0 else dec_inp
                    
                    # 1. Prepare encoder time marks (X_time)
                    if X_time is not None:
                        # Ensure X_time has the same sequence length as X
                        encoder_seq_len = X.size(1)
                        if X_time.size(1) != encoder_seq_len:
                            if X_time.size(1) < encoder_seq_len:
                                # Pad X_time to match X length
                                padding_len = encoder_seq_len - X_time.size(1)
                                last_time_entries = X_time[:, -1:, :].repeat(1, padding_len, 1)
                                enc_time_mark = torch.cat([X_time, last_time_entries], dim=1)
                            else:
                                # Slice X_time to match X length
                                enc_time_mark = X_time[:, :encoder_seq_len, :]
                        else:
                            enc_time_mark = X_time
                    else:
                        # Create dummy encoder time marks if none provided
                        enc_time_mark = torch.zeros((X.size(0), X.size(1), 5), device=self.device).long()
                    
                    # 2. Prepare decoder time marks
                    decoder_seq_len = dec_inp.size(1)  # This is label_len + pred_len
                    if Y_time is not None:
                        # Create a new time tensor with the correct sequence length
                        if Y_time.size(1) < decoder_seq_len:
                            # Pad Y_time to match dec_inp length
                            padding_len = decoder_seq_len - Y_time.size(1)
                            last_time_entries = Y_time[:, -1:, :].repeat(1, padding_len, 1)
                            dec_time_mark = torch.cat([Y_time, last_time_entries], dim=1)
                        else:
                            # Slice Y_time to match dec_inp length
                            dec_time_mark = Y_time[:, :decoder_seq_len, :]
                    else:
                        # Create dummy decoder time marks if none provided
                        dec_time_mark = torch.zeros((X.size(0), decoder_seq_len, 5), device=self.device).long()
                    
                    # Use X, enc_time_mark, dec_inp, dec_time_mark as inputs to the Informer model
                    predictions = self.model(X, enc_time_mark, dec_inp, dec_time_mark)
                else:
                    X, Y = batch[0], batch[1]
                    X = X.to(self.device)
                    Y = Y.to(self.device)
                    weights = batch[-1].to(self.device) if isinstance(batch[-1], torch.Tensor) else torch.ones_like(Y)
                    
                    # Forward pass - standard approach
                    predictions = self.model(X)
                
                if self.criterion == custom_loss:
                    loss = self.criterion(predictions, Y)
                else:
                    loss = self.criterion(predictions, Y)
                    # Fix the dimension mismatch in weights application
                    # First check tensor shapes
                    if weights.dim() == 1:
                        weights = weights.unsqueeze(1)
                    
                    # Get loss shape for proper broadcasting
                    loss_shape = loss.shape
                    
                    # Ensure weights have compatible dimensions with loss
                    if weights.size(0) == Y.size(0):  # Batch dimension matches
                        # Reshape weights to match loss dimensions for proper broadcasting
                        if loss.dim() > 1 and weights.dim() < loss.dim():
                            # Add missing dimensions to weights
                            for _ in range(loss.dim() - weights.dim()):
                                weights = weights.unsqueeze(-1)
                        
                        # Check if we need to expand weights along any dimension
                        if weights.shape != loss_shape:
                            try:
                                # Try to expand weights to match loss shape
                                expand_shape = []
                                for i, (w_size, l_size) in enumerate(zip(weights.shape, loss_shape)):
                                    if w_size == 1 and l_size > 1:
                                        expand_shape.append(l_size)
                                    else:
                                        expand_shape.append(w_size)
                                
                                if len(expand_shape) == len(weights.shape):
                                    weights = weights.expand(expand_shape)
                            except RuntimeError as e:
                                print(f"Warning: Could not reshape weights: {e}")
                                # If reshaping fails, use unweighted loss
                                loss = loss.mean()
                                continue
                    
                    # Apply weights only if dimensions are compatible
                    try:
                        if weights.shape == loss_shape or (weights.dim() == 1 and loss.dim() == 1):
                            loss = loss * weights
                        else:
                            # If shapes still don't match, just use mean of loss without weights
                            print(f"Warning: Weight shape {weights.shape} incompatible with loss shape {loss_shape}. Using unweighted loss.")
                    except RuntimeError as e:
                        print(f"Warning: Error applying weights: {e}. Using unweighted loss.")
                    
                    # Take mean of the weighted loss
                    loss = loss.mean()
                
                # Check for NaN loss
                if torch.isnan(loss).any():
                    print(f"Warning: NaN loss detected in validation. Using zero loss instead.")
                    loss_value = 0.0
                else:
                    loss_value = loss.item()
                
                total_loss += loss_value
                num_batches += 1
            
            avg_loss = total_loss / max(1, num_batches)
            self.metrics["val_loss"].append(avg_loss)
            
            return avg_loss


    def Test_Model(self, loader, save_path=None):
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_inputs = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Testing", leave=False):
                if self.is_informer and len(batch) > 4:  # Check if time data is included in the batch
                    # Unpack batch with time data
                    if len(batch) == 6:  # X, Y, X_time, Y_time, action, weight
                        X, Y, X_time, Y_time, actions, weights = batch
                    else:  # Fallback if batch structure is different
                        X, Y = batch[0], batch[1]
                        X_time, Y_time = batch[2], batch[3] if len(batch) > 3 else (None, None)
                        weights = batch[-1] if isinstance(batch[-1], torch.Tensor) else torch.ones_like(Y)
                    
                    X = X.to(self.device)
                    Y = Y.to(self.device)
                    X_time = X_time.to(self.device) if X_time is not None else None
                    Y_time = Y_time.to(self.device) if Y_time is not None else None
                    weights = weights.to(self.device)
                    
                    # Configure decoder input for Informer
                    label_len = self.lag // 2  # Half of lag is used as label portion
                    dec_inp = torch.zeros((X.size(0), self.n_ahead, X.size(2)), device=self.device)
                    dec_inp = torch.cat([Y[:, :label_len, :], dec_inp], dim=1) if label_len > 0 else dec_inp
                    
                    # 1. Prepare encoder time marks (X_time)
                    if X_time is not None:
                        # Ensure X_time has the same sequence length as X
                        encoder_seq_len = X.size(1)
                        if X_time.size(1) != encoder_seq_len:
                            if X_time.size(1) < encoder_seq_len:
                                # Pad X_time to match X length
                                padding_len = encoder_seq_len - X_time.size(1)
                                last_time_entries = X_time[:, -1:, :].repeat(1, padding_len, 1)
                                enc_time_mark = torch.cat([X_time, last_time_entries], dim=1)
                            else:
                                # Slice X_time to match X length
                                enc_time_mark = X_time[:, :encoder_seq_len, :]
                        else:
                            enc_time_mark = X_time
                    else:
                        # Create dummy encoder time marks if none provided
                        enc_time_mark = torch.zeros((X.size(0), X.size(1), 5), device=self.device).long()
                    
                    # 2. Prepare decoder time marks
                    decoder_seq_len = dec_inp.size(1)  # This is label_len + pred_len
                    if Y_time is not None:
                        # Create a new time tensor with the correct sequence length
                        if Y_time.size(1) < decoder_seq_len:
                            # Pad Y_time to match dec_inp length
                            padding_len = decoder_seq_len - Y_time.size(1)
                            last_time_entries = Y_time[:, -1:, :].repeat(1, padding_len, 1)
                            dec_time_mark = torch.cat([Y_time, last_time_entries], dim=1)
                        else:
                            # Slice Y_time to match dec_inp length
                            dec_time_mark = Y_time[:, :decoder_seq_len, :]
                    else:
                        # Create dummy decoder time marks if none provided
                        dec_time_mark = torch.zeros((X.size(0), decoder_seq_len, 5), device=self.device).long()
                    
                    # Use X, enc_time_mark, dec_inp, dec_time_mark as inputs to the Informer model
                    predictions = self.model(X, enc_time_mark, dec_inp, dec_time_mark)
                else:
                    X, Y = batch[0], batch[1]
                    X = X.to(self.device)
                    Y = Y.to(self.device)
                    weights = batch[-1].to(self.device) if isinstance(batch[-1], torch.Tensor) else torch.ones_like(Y)
                    
                    # Forward pass - standard approach
                    predictions = self.model(X)
                
                # Calculate loss
                if self.criterion == custom_loss:
                    loss = self.criterion(predictions, Y)
                else:
                    loss = self.criterion(predictions, Y)
                    # Fix the dimension mismatch in weights application
                    # First check tensor shapes
                    if weights.dim() == 1:
                        weights = weights.unsqueeze(1)
                    
                    # Get loss shape for proper broadcasting
                    loss_shape = loss.shape
                    
                    # Ensure weights have compatible dimensions with loss
                    if weights.size(0) == Y.size(0):  # Batch dimension matches
                        # Reshape weights to match loss dimensions for proper broadcasting
                        if loss.dim() > 1 and weights.dim() < loss.dim():
                            # Add missing dimensions to weights
                            for _ in range(loss.dim() - weights.dim()):
                                weights = weights.unsqueeze(-1)
                        
                        # Check if we need to expand weights along any dimension
                        if weights.shape != loss_shape:
                            try:
                                # Try to expand weights to match loss shape
                                expand_shape = []
                                for i, (w_size, l_size) in enumerate(zip(weights.shape, loss_shape)):
                                    if w_size == 1 and l_size > 1:
                                        expand_shape.append(l_size)
                                    else:
                                        expand_shape.append(w_size)
                                
                                if len(expand_shape) == len(weights.shape):
                                    weights = weights.expand(expand_shape)
                            except RuntimeError as e:
                                print(f"Warning: Could not reshape weights: {e}")
                                # If reshaping fails, use unweighted loss
                                loss = loss.mean()
                                continue
                    
                    # Apply weights only if dimensions are compatible
                    try:
                        if weights.shape == loss_shape or (weights.dim() == 1 and loss.dim() == 1):
                            loss = loss * weights
                        else:
                            # If shapes still don't match, just use mean of loss without weights
                            print(f"Warning: Weight shape {weights.shape} incompatible with loss shape {loss_shape}. Using unweighted loss.")
                    except RuntimeError as e:
                        print(f"Warning: Error applying weights: {e}. Using unweighted loss.")
                    
                    # Take mean of the weighted loss
                    loss = loss.mean()
                
                # Check for NaN loss
                if torch.isnan(loss).any():
                    print(f"Warning: NaN loss detected in testing. Using zero loss instead.")
                    loss_value = 0.0
                else:
                    loss_value = loss.item()
                
                total_loss += loss_value
                num_batches += 1
                
                # Store predictions and targets for metrics calculation
                all_predictions.append(predictions.detach().cpu())
                all_targets.append(Y.detach().cpu())
                all_inputs.append(X.detach().cpu())
            
            # Concatenate all batches
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            all_inputs = torch.cat(all_inputs, dim=0)
            
            # Calculate metrics
            mse = F.mse_loss(all_predictions, all_targets).item()
            mae = F.l1_loss(all_predictions, all_targets).item()
            
            # Calculate R^2 score
            target_mean = torch.mean(all_targets, dim=0)
            ss_tot = torch.sum((all_targets - target_mean) ** 2, dim=0)
            ss_res = torch.sum((all_targets - all_predictions) ** 2, dim=0)
            r2 = 1 - ss_res / (ss_tot + 1e-8)  # Add small epsilon to avoid division by zero
            r2 = torch.mean(r2).item()  # Average R^2 across all output dimensions
            
            avg_loss = total_loss / max(1, num_batches)
            
            # Print metrics
            print(f"Test Loss: {avg_loss:.6f}, MSE: {mse:.6f}, MAE: {mae:.6f}, R^2: {r2:.6f}")
            
            # Save results if path provided
            if save_path:
                results = {
                    "loss": avg_loss,
                    "mse": mse,
                    "mae": mae,
                    "r2": r2,
                    "predictions": all_predictions.numpy(),
                    "targets": all_targets.numpy(),
                    "inputs": all_inputs.numpy()
                }
                torch.save(results, save_path)
            
            return {"loss": avg_loss, "mse": mse, "mae": mae, "r2": r2}


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

        # Define lag labels
        lag_labels = [f"Lag {i}" for i in range(n_channels_lag)]

        # Use sequential timesteps - ensure x_forecast has the right length
        x_lag = np.arange(0, n_lag)
        x_forecast = np.arange(n_lag, n_lag + Y_sample.shape[0])

        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot all lag inputs with proper labels
        for ch in range(n_channels_lag):
            ax.plot(x_lag, X_sample[:, ch], marker='o', label=f"{lag_labels[ch]} (Input)")

        # Plot all outputs/predictions with proper names
        for ch in range(n_channels_to_plot):
            out_name = f"Output {ch}"
            
            # Plot ground truth
            ax.plot(x_forecast, Y_sample[:, ch], marker='o', label=f"{out_name} (GT)")
            
            # Ensure Y_pred has the same length as x_forecast for plotting
            if Y_pred.shape[0] != len(x_forecast):
                if Y_pred.shape[0] > len(x_forecast):
                    # Truncate Y_pred to match x_forecast length
                    Y_pred_plot = Y_pred[:len(x_forecast), ch]
                else:
                    # Pad Y_pred with the last value to match x_forecast length
                    padding = np.full(len(x_forecast) - Y_pred.shape[0], Y_pred[-1, ch])
                    Y_pred_plot = np.concatenate([Y_pred[:, ch], padding])
            else:
                Y_pred_plot = Y_pred[:, ch]
                
            # Plot prediction
            ax.plot(x_forecast, Y_pred_plot, marker='x', linestyle='--', label=f"{out_name} (Pred)")

        # Draw a vertical line at the transition from input to prediction
        ax.axvline(x=n_lag, color='gray', linestyle=':', linewidth=1)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Signal Value")

        title = "Prediction vs Ground Truth"
        if action is not None:
            title += f" - Action {action}"
        ax.set_title(title)

        # Use a smaller font size and move legend outside plot for better visibility
        ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close(fig)

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
                if self.is_informer and len(batch) > 4:  # Check if time data is included in the batch
                    # Unpack batch with time data
                    if len(batch) == 6:  # X, Y, X_time, Y_time, action, weight
                        X, Y, X_time, Y_time, actions, weights = batch
                    else:  # Fallback if batch structure is different
                        X, Y = batch[0], batch[1]
                        X_time, Y_time = batch[2], batch[3] if len(batch) > 3 else (None, None)
                        actions = batch[4] if len(batch) > 4 else ["unknown"] * X.size(0)
                        weights = batch[-1] if isinstance(batch[-1], torch.Tensor) else torch.ones_like(Y)
                    
                    X = X.to(self.device)
                    Y = Y.to(self.device)
                    X_time = X_time.to(self.device) if X_time is not None else None
                    Y_time = Y_time.to(self.device) if Y_time is not None else None
                    
                    # Process each sample in the batch
                    for i in range(X.size(0)):
                        action = actions[i] if isinstance(actions, list) else "unknown"
                        
                        # Skip if we already have enough samples for this action
                        if action_counts.get(action, 0) >= num_windows:
                            continue
                        
                        # Create action-specific directory
                        action_dir = os.path.join(save_dir, f"action_{action}")
                        os.makedirs(action_dir, exist_ok=True)
                        
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
                                padding = torch.zeros((dec_inp.size(1) - Y_time[i].size(0), 5), device=self.device).long()
                                y_time_padded = torch.cat([Y_time[i], padding], dim=0)
                                x_mark_dec = y_time_padded.unsqueeze(0)
                            else:
                                x_mark_dec = Y_time[i, :dec_inp.size(1), :].unsqueeze(0)
                        
                        # Get prediction from Informer model
                        prediction = self.model(x_enc, x_mark_enc, dec_inp, x_mark_dec).squeeze(0)
                        
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
                else:
                    X, Y = batch[0], batch[1]
                    X = X.to(self.device)
                    Y = Y.to(self.device)
                    
                    # Get actions from batch if available
                    if len(batch) > 2 and isinstance(batch[2], list):
                        actions = batch[2]
                    else:
                        actions = ["unknown"] * X.size(0)
                    
                    # Process each sample in the batch
                    for i in range(X.size(0)):
                        action = actions[i] if isinstance(actions, list) else "unknown"
                        
                        # Skip if we already have enough samples for this action
                        if action_counts.get(action, 0) >= num_windows:
                            continue
                        
                        # Create action-specific directory
                        action_dir = os.path.join(save_dir, f"action_{action}")
                        os.makedirs(action_dir, exist_ok=True)
                        
                        # Plot and save
                        save_path = os.path.join(action_dir, f"val_sample_{action_counts.get(action, 0)+1}.png")
                        self.plot_prediction(
                            X[i].detach().cpu().numpy(),
                            Y[i].detach().cpu().numpy(),
                            self.model(X[i].unsqueeze(0)).squeeze(0).detach().cpu().numpy(),
                            save_path,
                            action=action
                        )
                        
                        action_counts[action] = action_counts.get(action, 0) + 1

    def plot_test_results(self, test_loader, save_dir, num_windows=3):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Testing and saving results to: {save_dir}")
        
        self.model.eval()
        
        action_counts = {}
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing", leave=False):
                if self.is_informer and len(batch) > 4:  # Check if time data is included in the batch
                    # Unpack batch with time data
                    if len(batch) == 6:  # X, Y, X_time, Y_time, action, weight
                        X, Y, X_time, Y_time, actions, weights = batch
                    else:  # Fallback if batch structure is different
                        X, Y = batch[0], batch[1]
                        X_time, Y_time = batch[2], batch[3] if len(batch) > 3 else (None, None)
                        actions = batch[4] if len(batch) > 4 else ["unknown"] * X.size(0)
                        weights = batch[-1] if isinstance(batch[-1], torch.Tensor) else torch.ones_like(Y)
                    
                    X = X.to(self.device)
                    Y = Y.to(self.device)
                    X_time = X_time.to(self.device) if X_time is not None else None
                    Y_time = Y_time.to(self.device) if Y_time is not None else None
                    
                    # Process each sample in the batch
                    for i in range(X.size(0)):
                        action = actions[i] if isinstance(actions, list) else "unknown"
                        
                        # Skip if we already have enough samples for this action
                        if action_counts.get(action, 0) >= num_windows:
                            continue
                        
                        # Create action-specific directory
                        action_dir = os.path.join(save_dir, f"action_{action}")
                        os.makedirs(action_dir, exist_ok=True)
                        
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
                                padding = torch.zeros((dec_inp.size(1) - Y_time[i].size(0), 5), device=self.device).long()
                                y_time_padded = torch.cat([Y_time[i], padding], dim=0)
                                x_mark_dec = y_time_padded.unsqueeze(0)
                            else:
                                x_mark_dec = Y_time[i, :dec_inp.size(1), :].unsqueeze(0)
                        
                        # Get prediction from Informer model
                        prediction = self.model(x_enc, x_mark_enc, dec_inp, x_mark_dec).squeeze(0)
                        
                        # Plot and save
                        save_path = os.path.join(action_dir, f"test_sample_{action_counts.get(action, 0)+1}.png")
                        self.plot_prediction(
                            X[i].detach().cpu().numpy(),
                            Y[i].detach().cpu().numpy(),
                            prediction.detach().cpu().numpy(),
                            save_path,
                            action=action
                        )
                        
                        action_counts[action] = action_counts.get(action, 0) + 1
                else:
                    X = batch[0].to(self.device)  # Move X to the correct device
                    Y = batch[1]
                    # Check if actions are provided in the batch
                    actions = batch[2] if len(batch) > 2 else torch.zeros(X.size(0), dtype=torch.long)
                
                    batch_size = Y.size(0)
                    
                    for i in range(batch_size):
                        action = actions[i].item() if isinstance(actions[i], torch.Tensor) else actions[i]
                        
                        # Skip if we already have enough windows for this action
                        if action_counts.get(action, 0) >= num_windows:
                            continue
                        
                        # Create action-specific directory
                        action_dir = os.path.join(save_dir, f"action_{action}")
                        os.makedirs(action_dir, exist_ok=True)
                        
                        # Plot and save
                        save_path = os.path.join(action_dir, f"test_sample_{action_counts.get(action, 0)+1}.png")
                        self.plot_prediction(
                            X[i].detach().cpu().numpy(),
                            Y[i].detach().cpu().numpy(),
                            self.model(X[i].unsqueeze(0)).squeeze(0).detach().cpu().numpy(),
                            save_path,
                            action=action
                        )
                        
                        action_counts[action] = action_counts.get(action, 0) + 1
                
                # Check if we have enough samples for all actions
                if action_counts and all(count >= num_windows for count in action_counts.values()):
                    break
        
        print(f"Test results saved to {save_dir}")

    def save_first_windows(self, train_loader, epoch, save_dir, num_windows=5):
        os.makedirs(save_dir, exist_ok=True)
        
        self.model.eval()
        
        action_counts = {}
        
        with torch.no_grad():
            for batch in train_loader:
                if self.is_informer and len(batch) > 4:  # Check if time data is included in the batch
                    # Unpack batch with time data
                    if len(batch) == 6:  # X, Y, X_time, Y_time, action, weight
                        X, Y, X_time, Y_time, actions, weights = batch
                    else:  # Fallback if batch structure is different
                        X, Y = batch[0], batch[1]
                        X_time, Y_time = batch[2], batch[3] if len(batch) > 3 else (None, None)
                        actions = batch[4] if len(batch) > 4 else ["unknown"] * X.size(0)
                        weights = batch[-1] if isinstance(batch[-1], torch.Tensor) else torch.ones_like(Y)
                    
                    X = X.to(self.device)
                    Y = Y.to(self.device)
                    X_time = X_time.to(self.device) if X_time is not None else None
                    Y_time = Y_time.to(self.device) if Y_time is not None else None
                    
                    # Process each sample in the batch
                    for i in range(X.size(0)):
                        action = actions[i] if isinstance(actions, list) else "unknown"
                        
                        # Skip if we already have enough samples for this action
                        if action_counts.get(action, 0) >= num_windows:
                            continue
                        
                        # Create action-specific directory
                        action_dir = os.path.join(save_dir, f"action_{action}")
                        os.makedirs(action_dir, exist_ok=True)
                        
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
                                padding = torch.zeros((dec_inp.size(1) - Y_time[i].size(0), 5), device=self.device).long()
                                y_time_padded = torch.cat([Y_time[i], padding], dim=0)
                                x_mark_dec = y_time_padded.unsqueeze(0)
                            else:
                                x_mark_dec = Y_time[i, :dec_inp.size(1), :].unsqueeze(0)
                        
                        # Get prediction from Informer model
                        prediction = self.model(x_enc, x_mark_enc, dec_inp, x_mark_dec).squeeze(0)
                        
                        # Plot and save
                        save_path = os.path.join(action_dir, f"epoch_{epoch}_sample_{action_counts.get(action, 0)+1}.png")
                        self.plot_prediction(
                            X[i].detach().cpu().numpy(),
                            Y[i].detach().cpu().numpy(),
                            prediction.detach().cpu().numpy(),
                            save_path,
                            action=action
                        )
                        
                        action_counts[action] = action_counts.get(action, 0) + 1
                else:
                    X, Y = batch[0], batch[1]
                    X = X.to(self.device)
                    Y = Y.to(self.device)
                    
                    # Get actions from batch if available
                    if len(batch) > 2 and isinstance(batch[2], list):
                        actions = batch[2]
                    else:
                        actions = ["unknown"] * X.size(0)
                    
                    # Process each sample in the batch
                    for i in range(X.size(0)):
                        action = actions[i] if isinstance(actions, list) else "unknown"
                        
                        # Skip if we already have enough samples for this action
                        if action_counts.get(action, 0) >= num_windows:
                            continue
                        
                        # Create action-specific directory
                        action_dir = os.path.join(save_dir, f"action_{action}")
                        os.makedirs(action_dir, exist_ok=True)
                        
                        # Plot and save
                        save_path = os.path.join(action_dir, f"epoch_{epoch}_sample_{action_counts.get(action, 0)+1}.png")
                        self.plot_prediction(
                            X[i].detach().cpu().numpy(),
                            Y[i].detach().cpu().numpy(),
                            self.model(X[i].unsqueeze(0)).squeeze(0).detach().cpu().numpy(),
                            save_path,
                            action=action
                        )
                        
                        action_counts[action] = action_counts.get(action, 0) + 1
                
                # Check if we have enough samples for all actions
                if action_counts and all(count >= num_windows for count in action_counts.values()):
                    break

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
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Loss curve saved to {save_path}")

    def step_scheduler(self):
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # ReduceLROnPlateau requires a metrics value
            self.scheduler.step(self.metrics["val_loss"][-1])
        else:
            # Other schedulers like StepLR, CosineAnnealingLR don't need metrics
            self.scheduler.step()
    
    def log_epoch_loss(self, epoch, train_loss, val_loss, log_file_path):
        with open(log_file_path, 'a') as f:
            f.write(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}\n")
    
    def fit(self, train_loader, val_loader, epochs, checkpoint_dir, patience=10,
            min_delta=0.0, num_windows=10, loss_curve_path='loss_curve.png', start_epoch=0):
        """
        Train the model with early stopping.
        
        Parameters:
        - epochs: Total number of epochs to train
        - checkpoint_dir: Directory to save model checkpoints
        - patience: Number of epochs to wait before early stopping
        - min_delta: Minimum change in validation loss to qualify as improvement
        - num_windows: Number of windows to plot for visualization
        - start_epoch: Starting epoch when resuming training (default: 1)
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create figures directory
        figures_dir = os.path.join(os.path.dirname(checkpoint_dir), "figures")
        os.makedirs(figures_dir, exist_ok=True)
        
        log_file_path = os.path.join(checkpoint_dir, "training_log.txt")
        early_stopping = EarlyStopping(
            patience=patience, 
            min_delta=min_delta, 
            verbose=True, 
            path=os.path.join(checkpoint_dir, "best_model.pt"),
            start_epoch=start_epoch
        )
        
        for epoch in range(start_epoch, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss = self.val_epoch(val_loader)
            
            # Log loss
            self.log_epoch_loss(epoch, train_loss, val_loss, log_file_path)
            
            # Create epoch-specific figure directory under figures_dir
            epoch_fig_dir = os.path.join(figures_dir, f"epoch_{epoch}")
            os.makedirs(epoch_fig_dir, exist_ok=True)
            
            # Save 5 windows per action for training data
            self.save_first_windows(train_loader, epoch, epoch_fig_dir, num_windows=5)
            
            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            
            # Early stopping check
            if early_stopping(val_loss, self.model, epoch):
                print(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Plot loss curve
            self.plot_loss_curve(os.path.join(figures_dir, "loss_curve.png"))
            
            # Step the scheduler
            self.step_scheduler()
        
        # Load the best model
        self.model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_model.pt")))
        
        # Plot validation results - 3 windows per action
        val_plot_dir = os.path.join(figures_dir, "validation")
        os.makedirs(val_plot_dir, exist_ok=True)
        self.plot_validation_results(val_loader, val_plot_dir, num_windows=3)
        
        # Plot test results - 3 windows per action
        test_plot_dir = os.path.join(figures_dir, "test")
        os.makedirs(test_plot_dir, exist_ok=True)
        self.plot_test_results(self.testloader, test_plot_dir, num_windows=3)
        
        return self.Metrics
