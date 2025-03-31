import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

# Example: MDN loss function.
def mdn_loss(pi, sigma, mu, target, eps=1e-6):
    """
    Computes the negative log likelihood for a Mixture Density Network.
    
    Args:
        pi (Tensor): Mixture weights, shape [batch, n_ahead, num_mixtures]
        sigma (Tensor): Standard deviations, shape [batch, n_ahead, num_mixtures]
        mu (Tensor): Means, shape [batch, n_ahead, num_mixtures, num_channels]
        target (Tensor): Ground truth, shape [batch, n_ahead, num_channels]
        eps (float): Small constant for numerical stability.
    Returns:
        Tensor: Negative log likelihood loss (scalar)
    """
    # Expand target to shape [batch, n_ahead, num_mixtures, num_channels]
    target_exp = target.unsqueeze(2).expand_as(mu)
    # Compute the probability of target under each Gaussian.
    exponent = -0.5 * ((target_exp - mu) / (sigma.unsqueeze(-1) + eps)) ** 2
    gaussian = (1.0 / (sigma.unsqueeze(-1) * np.sqrt(2 * np.pi) + eps)) * torch.exp(exponent)
    # Multiply by mixture weights.
    weighted = pi.unsqueeze(-1) * gaussian
    # Sum over mixtures.
    prob = weighted.sum(dim=2) + eps  # shape: [batch, n_ahead, num_channels]
    # Negative log likelihood
    nll = -torch.log(prob)
    return nll.mean()

class Trainer:
    def __init__(self, model, optimizer, scheduler, testloader, fig_dir, loss, model_type="lstm", device="cpu",
                 is_classification=False, use_variation_penalty=True, alpha=1.0, var_threshold=0.01, **kwargs):
        """
        Args:
            model: The PyTorch model.
            optimizer: The optimizer.
            scheduler: A learning rate scheduler.
            testloader: DataLoader for test data.
            fig_dir: Base directory for saving figures.
            loss: A string identifier for loss type ("mse", "smoothl1", "huber", "mdn").
            model_type: A string identifier (e.g., "lstm", "tcn", "mdn", etc.).
            device: "cpu" or "cuda".
            is_classification: Boolean flag. If True, classification metrics are used.
            use_variation_penalty (bool): Whether to use the variation penalty.
            alpha (float): Weight for the penalty term.
            var_threshold (float): Minimum acceptable variance.
            kwargs: Other optional keyword arguments.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.testloader = testloader
        self.fig_dir = fig_dir
        self.model_type = model_type.lower()
        self.device = device
        self.is_classification = is_classification

        self.use_variation_penalty = use_variation_penalty
        self.alpha = alpha
        self.var_threshold = var_threshold

        # Choose base loss function for non-MDN cases.
        if loss == "mse":
            self.loss = nn.MSELoss
        elif loss == "smoothl1":
            self.loss = nn.SmoothL1Loss
        elif loss == "huber":
            self.loss = nn.HuberLoss
        elif loss == "mdn":
            # For MDN, we won't use the standard criterion.
            self.loss = None
        else:
            raise ValueError("Invalid loss type provided.")
        # Set reduction="none" to compute per-sample losses when applicable.
        if self.loss is not None:
            self.criterion = self.loss(reduction="none")
        else:
            self.criterion = None

        # Store metrics.
        if self.is_classification:
            self.metrics = {
                "train_loss": [],
                "val_loss": [],
                "train_acc": [],
                "val_acc": []
            }
        else:
            self.metrics = {
                "train_loss": [],
                "val_loss": []
            }
        
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

    def custom_loss(self, preds, targets, aux_preds, bin_weight, 
                    lambda_weight=2.0, desired_variation=0.01, beta=1.0,
                    freq_weight=1.0, aux_weight=1.0):
        # Reshape bin_weight for broadcasting.
        bin_weight = bin_weight.view(-1, 1, 1)
        
        # Weighted MSE term.
        importance = bin_weight * (1.0 + lambda_weight * torch.abs(targets))
        weighted_mse = importance * (preds - targets) ** 2
        base_loss = weighted_mse.mean()
        
        # Temporal variation penalty.
        temporal_diff = torch.abs(preds[:, 1:, :] - preds[:, :-1, :])
        variation_penalty = torch.clamp(desired_variation - temporal_diff.mean(), min=0)
        
        # Frequency domain loss.
        fft_preds = torch.fft.rfft(preds, dim=1)
        fft_targets = torch.fft.rfft(targets, dim=1)
        freq_loss = F.mse_loss(torch.abs(fft_preds), torch.abs(fft_targets))
        
        # Auxiliary loss.
        true_diff = targets[:, 1:, :] - targets[:, :-1, :]
        aux_loss = F.mse_loss(aux_preds[:, 1:, :], true_diff)
        
        total_loss = base_loss + beta * variation_penalty + freq_weight * freq_loss + aux_weight * aux_loss
        
        # Log each term (you might use print statements, or better, log to a file/metric dictionary)
        self.last_loss_terms = {
            "base_loss": base_loss.item(),
            "variation_penalty": variation_penalty.item(),
            "freq_loss": freq_loss.item(),
            "aux_loss": aux_loss.item(),
            "total_loss": total_loss.item()
        }
        
        return total_loss


    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        count = 0
        for batch in tqdm(dataloader, desc="Training", leave=False):
            # Unpack batch. For regression, expect (X, y, action, target_leg, weight)
            X, y, _, _, weight = batch  
            X = X.to(self.device)
            y = y.to(self.device)
            weight = weight.to(self.device)  # shape: [batch]
            
            self.optimizer.zero_grad()
            
            # Forward pass. For MDN models, assume the model returns a dict.
            outputs = self.model(X)
            if self.model_type == "mdn":
                # Assume outputs is a dict with keys: "pi", "sigma", "mu"
                loss_value = mdn_loss(outputs["pi"], outputs["sigma"], outputs["mu"], y)
            else:
                # For models with auxiliary branch, assume the model returns (main, aux)
                if isinstance(outputs, (list, tuple)):
                    main_preds, aux_preds = outputs
                else:
                    # If no auxiliary output, set aux_preds to zeros.
                    main_preds = outputs
                    aux_preds = torch.zeros_like(outputs)
                loss_value = self.custom_loss(main_preds, y, aux_preds, weight)
                
                # Inside your training loop:
               
            loss_value.backward()
            self.optimizer.step()

            total_loss += loss_value.item() * X.size(0)
            count += X.size(0)

        avg_loss = total_loss / count
        if hasattr(self, "last_loss_terms"):
            print("Loss breakdown:", self.last_loss_terms)
        self.metrics["train_loss"].append(avg_loss)
        
    
    def val_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0
        count = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation", leave=False):
                X, y, _, _, weight = batch
                X = X.to(self.device)
                y = y.to(self.device)
                weight = weight.to(self.device)

                outputs = self.model(X)
                if self.model_type == "mdn":
                    loss_value = mdn_loss(outputs["pi"], outputs["sigma"], outputs["mu"], y)
                else:
                    if isinstance(outputs, (list, tuple)):
                        main_preds, aux_preds = outputs
                    else:
                        main_preds = outputs
                        aux_preds = torch.zeros_like(outputs)
                    loss_value = self.custom_loss(main_preds, y, aux_preds, weight)
                
                total_loss += loss_value.item() * X.size(0)
                count += X.size(0)

        avg_loss = total_loss / count
        self.metrics["val_loss"].append(avg_loss)
        
    def Training_Loop(self, dataloader):
        self.train_epoch(dataloader)
        
    def Validation_Loop(self, dataloader):
        self.val_epoch(dataloader)
        
    def step_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()
        
    def Test_Model(self, test_loader=None):
        if test_loader is None:
            test_loader = self.testloader
        
        self.model.eval()
        total_loss = 0
        count = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in test_loader:
                X, y = batch[0], batch[1]
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                if self.model_type == "mdn":
                    loss = mdn_loss(outputs["pi"], outputs["sigma"], outputs["mu"], y)
                else:
                    if isinstance(outputs, (list, tuple)):
                        main_preds, _ = outputs
                    else:
                        main_preds = outputs
                    loss = self.criterion(main_preds, y)
                total_loss += loss.item() * X.size(0)
                all_preds.append(outputs if self.model_type != "mdn" else outputs["mu"])
                all_targets.append(y)
                count += X.size(0)
        
        avg_loss = total_loss / count
        print(f"Test Loss = {avg_loss}")
        self.test_results = {"loss": avg_loss}
        self.test_results["preds"] = torch.cat(all_preds, dim=0)
        self.test_results["targets"] = torch.cat(all_targets, dim=0)
    
    def plot_predictions(self, epoch):
        os.makedirs(self.fig_dir, exist_ok=True)
        save_path = os.path.join(self.fig_dir, f"predictions_epoch_{epoch+1}.png")
        self.model.eval()
        with torch.no_grad():
            batch = next(iter(self.testloader))
            X, y = batch[0], batch[1]
            X = X.to(self.device)
            outputs = self.model(X)
            if self.model_type == "mdn":
                preds = outputs["mu"]
            elif isinstance(outputs, (list, tuple)):
                preds, _ = outputs
            else:
                preds = outputs
            X_cpu = X.detach().cpu().numpy()
            preds_cpu = preds.detach().cpu().numpy()
            y_cpu = y.detach().cpu().numpy()
            sample_idx = 0
            plt.figure(figsize=(10, 6))
            if not self.is_classification:
                n_channels = min(3, y_cpu[sample_idx].shape[1])
                for c in range(n_channels):
                    plt.plot(y_cpu[sample_idx][:, c], label=f"GT Channel {c}")
                for c in range(n_channels):
                    plt.plot(preds_cpu[sample_idx][:, c], linestyle="--", label=f"Pred Channel {c}")
                plt.title(f"Epoch {epoch+1} - Sample Predictions")
            else:
                window_data = X_cpu[sample_idx]
                if window_data.ndim == 2:
                    seq_len, n_channels = window_data.shape
                    for c in range(n_channels):
                        plt.plot(window_data[:, c], label=f"Channel {c}")
                else:
                    plt.plot(window_data)
                predicted_class = int(np.argmax(preds_cpu[sample_idx]))
                actual_class = int(np.argmax(y_cpu[sample_idx])) if isinstance(y_cpu[sample_idx], np.ndarray) and y_cpu[sample_idx].ndim > 0 else int(y_cpu[sample_idx])
                plt.title(f"Epoch {epoch+1}\nPredicted: {predicted_class}, Ground Truth: {actual_class}")
            plt.legend()
            plt.savefig(save_path)
            plt.close()
            print(f"Saved predictions plot to {save_path}")
            
    def plot_validation_results(self, val_loader, save_path):
        self.model.eval()
        with torch.no_grad():
            batch = next(iter(val_loader))
            X, y = batch[0], batch[1]
            X = X.to(self.device)
            outputs = self.model(X)
            if self.model_type == "mdn":
                preds = outputs["mu"]
            elif isinstance(outputs, (list, tuple)):
                preds, _ = outputs
            else:
                preds = outputs
            X_cpu = X.detach().cpu().numpy()
            preds_cpu = preds.detach().cpu().numpy()
            y_cpu = y.detach().cpu().numpy()
            sample_idx = 0
            channel = 0
            combined_true = np.concatenate([X_cpu[sample_idx][:, channel], y_cpu[sample_idx][:, channel]])
            combined_pred = np.concatenate([X_cpu[sample_idx][:, channel], preds_cpu[sample_idx][:, channel]])
            plt.figure(figsize=(10, 6))
            plt.plot(combined_true, label="Ground Truth", marker="o")
            plt.plot(combined_pred, label="Prediction", marker="x", linestyle="--")
            plt.axvline(x=len(X_cpu[sample_idx])-0.5, color="black", linestyle="--", label="Forecast Boundary")
            plt.xlabel("Time Step")
            plt.ylabel("Signal Value")
            plt.title("Validation Combined Prediction")
            plt.legend()
            plt.savefig(save_path)
            plt.close()
            print(f"Saved validation results plot to {save_path}")
            
    def save_first_10_windows(self, train_loader, epoch):
        save_dir = os.path.join(self.fig_dir, self.model_type, f"epoch_{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving sample windows to: {save_dir}")
        self.model.eval()
        saved_count = 0
        with torch.no_grad():
            for batch in train_loader:
                X_batch, y_batch = batch[0], batch[1]
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                if self.model_type == "mdn":
                    preds = outputs["mu"]
                elif isinstance(outputs, (list, tuple)):
                    preds, _ = outputs
                else:
                    preds = outputs
                X_batch_cpu = X_batch.detach().cpu().numpy()
                preds_cpu = preds.detach().cpu().numpy()
                y_batch_cpu = y_batch.detach().cpu().numpy()
                batch_size = X_batch_cpu.shape[0]
                for i in range(batch_size):
                    if saved_count >= 10:
                        break
                    plt.figure()
                    if not self.is_classification:
                        n_channels = min(3, y_batch_cpu[i].shape[1])
                        for c in range(n_channels):
                            plt.plot(y_batch_cpu[i][:, c], label=f"GT Channel {c}")
                        for c in range(n_channels):
                            plt.plot(preds_cpu[i][:, c], linestyle="--", label=f"Pred Channel {c}")
                        plt.title(f"Epoch {epoch} - Window {saved_count}")
                        plt.legend()
                    else:
                        window_data = X_batch_cpu[i]
                        if window_data.ndim == 2:
                            seq_len, n_channels = window_data.shape
                            for c in range(n_channels):
                                plt.plot(window_data[:, c], label=f"Channel {c}")
                        else:
                            plt.plot(window_data)
                        predicted_class = int(np.argmax(preds_cpu[i]))
                        actual_class = int(np.argmax(y_batch_cpu[i])) if isinstance(y_batch_cpu[i], np.ndarray) and y_batch_cpu[i].ndim > 0 else int(y_batch_cpu[i])
                        plt.title(f"Epoch {epoch} - Window {saved_count}\nPred: {predicted_class}, GT: {actual_class}")
                        plt.legend()
                    save_path = os.path.join(save_dir, f"window_{saved_count}.png")
                    plt.savefig(save_path)
                    plt.close()
                    saved_count += 1
                if saved_count >= 10:
                    break
