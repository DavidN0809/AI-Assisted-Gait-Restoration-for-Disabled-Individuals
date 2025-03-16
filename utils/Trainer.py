from torchmetrics import ConfusionMatrix
from matplotlib import pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import copy
import torch

class EarlyStopping():
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
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
    def __init__(self, model, loss, optimizer, accuracy, model_type, device,
                 classes=0, noPrint=False, flatten_output=False):
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
        self.model_type = model_type
        self.classNum = classes
        self.noPrint = noPrint
        self.flatten_output = flatten_output
        
        # For flattening outputs if needed
        if flatten_output:
            self.flat = nn.Flatten()
        
        if noPrint:
            self.progressbar = lambda x: x
        else:
            self.progressbar = tqdm
        
        # Dictionary to store metrics over epochs.
        # For classification, we track accuracy and confusion matrix.
        if model_type == "Classification":
            self.Metrics = {"Training Loss": [], "Validation Loss": [],
                            "Training Accuracy": [], "Validation Accuracy": [],
                            "Test Loss": 0, "Test Accuracy": 0}
            self.ConfMatrix = None
        else:
            self.Metrics = {"Training Loss": [], "Validation Loss": [], "Test Loss": 0}
    
    def Training_Loop(self, Loader):
        self.model.train()
        tLossSum = 0
        tAccuracy = 0
        
        for data, labels in self.progressbar(Loader):
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
                # Convert predictions and labels back to integers for accuracy.
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
            for data, labels in Loader:
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
   
    def fit(self, trainingLoader, validateLoader, EPOCHS, start_epoch=0):
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
            if ES(self.model, self.Metrics["Validation Loss"][-1]):
                if not self.noPrint:
                    print("Stopping Model Early:", ES.status)
                break

    
    def Test_Model(self, testLoader):
        self.model.eval()
        total_loss = 0
        
        # For classification, set up confusion matrix and accuracy.
        if self.model_type == "Classification":
            confusion = ConfusionMatrix(task="multiclass", num_classes=self.classNum).to(self.device)
            total_accuracy = 0
        
        with torch.no_grad():
            for data, labels in testLoader:
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
                    
                total_loss += loss_val.item()
                
                if self.model_type == "Classification":
                    pred_labels = torch.tensor([torch.argmax(i).item() for i in pred]).to(self.device)
                    true_labels = torch.tensor([torch.argmax(i).item() for i in labels]).to(self.device)
                    total_accuracy += self.accuracy(pred_labels, true_labels)
                    confusion.update(pred_labels, true_labels)
        
        self.Metrics["Test Loss"] = total_loss / len(testLoader)
        if self.model_type == "Classification":
            self.Metrics["Test Accuracy"] = total_accuracy / len(testLoader)
            self.ConfMatrix = confusion.compute().cpu()
        
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
            
            # Optionally, plot the confusion matrix if available.
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
            plt.close()  # Close the current figure to free up memory
        else:
            plt.show()


    def reset(self):
        if self.model_type == "Classification":
            self.Metrics = {"Training Loss": [], "Validation Loss": [],
                            "Training Accuracy": [], "Validation Accuracy": [],
                            "Test Loss": 0, "Test Accuracy": 0}
            self.ConfMatrix = None
        else:
            self.Metrics = {"Training Loss": [], "Validation Loss": [], "Test Loss": 0}
