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
        if self.best_loss == None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model)
            
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model.load_state_dict(model.state_dict())
            
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.status = f"Stopped on {self.counter}"
            
            if self.restore_best_weights:
                model.load_state_dict(self.best_model.state_dict())
            return True
        
        self.status = f"{self.counter}/{self.patience}"
        return False


# Creates a large trainer class for training and saving metrics of our model
class ModelTrainer:
    
    def __init__(self, model, loss, optimizer, accuracy, model_type, device, classes=0, noPrint=False):
        
        # Sets model to GPU and basic loss function and optimizer used
        self.device = device
        self.model = model.to(device)
        self.Loss_Function = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
        self.model_type = model_type
        self.classNum = classes
        self.noPrint = noPrint
        self.flat = nn.Flatten()
        
        if noPrint:
            self.progressbar = lambda x: x
        else:
            self.progressbar = tqdm
        
        # Place to store metrics of our model throughout training and testing
        self.Metrics = {"Training Loss":[], "Validation Loss":[], 
                        "Training Accuracy":[], "Validation Accuracy":[],
                        "Test Accuracy":0, "Test Loss":0} 
        
        # Place to save confidence matrix 
        self.ConfMatrix = None
    
    # Defines the training loop for training our model
    def Training_Loop(self, Loader):
        
        # Sets model into training mode
        self.model.train()
        
        # Sets up metric grabing and an accuracy function
        if self.model_type == "Classification":
            MCA = self.accuracy(self.classNum)
        else:
            MCA = self.accuracy
        
        tLossSum = 0
        tAccuracy = 0
        
        # Iterates through dataloader
        for data, labels in self.progressbar(Loader):
            
            # Moves inputs and outputs to GPU and makes the labels one-hot vectors
            data = data.to(self.device)
            if self.model_type == "Classification":
                labels = torch.eye(10)[labels]
            # else:
            #     labels = labels.reshape(-1, 1)
            labels = labels.to(self.device)
            
            
            # Model makes prediction which is passed into a loss function
            pred = self.model(data)
            loss_val = self.Loss_Function(self.flat(pred), self.flat(labels))
            
            # Backpropagation model etc, etc...
            self.optimizer.zero_grad()
            loss_val.backward()
            self.optimizer.step()
            
            
            # Set the predictions and labels back into integers for accuracy calculation
            if self.model_type == "Classification":
                pred = torch.Tensor([torch.argmax(i).item() for i in pred])
                labels = torch.Tensor([torch.argmax(i).item() for i in labels])
            
            # Running Loss and accuracy
            tLossSum += loss_val.item()
            if self.model_type == "Classification":
                tAccuracy += MCA(pred, labels)
        
        # Update metrics based on running loss and accuracy
        self.Metrics["Training Loss"].append(tLossSum / len(Loader))
        if self.model_type == "Classification":
            self.Metrics["Training Accuracy"].append(tAccuracy / len(Loader))
        
        
    # Defines a function for validating our model is generalizing
    def Validation_Loop(self, Loader):
        
        # Sets model into evaluation mode
        self.model.eval()
        
        # Sets up metric grabing and an accuracy function
        if self.model_type == "Classification":
            MCA = self.accuracy(self.classNum)
        else:
            MCA = self.accuracy
            
        tLossSum = 0
        tAccuracy = 0
        
        # Iterates through dataloader
        for data, labels in Loader:
            
            # Moves inputs and outputs to GPU and makes the labels one-hot vectors
            data = data.to(self.device)
            if self.model_type == "Classification":
                labels = torch.eye(10)[labels]
            # else:
            #     labels = labels.reshape(-1, 1)
            labels = labels.to(self.device)
            
            # No Backpropagation, use no_grad to get simple prediction and loss
            with torch.no_grad():
                pred = self.model(data)
            loss_val = self.Loss_Function(self.flat(pred), self.flat(labels))
            
            # Set the predictions and labels back into integers for accuracy calculation
            if self.model_type == "Classification":
                pred = torch.Tensor([torch.argmax(i).item() for i in pred])
                labels = torch.Tensor([torch.argmax(i).item() for i in labels])
            
            # Running Loss and accuracy
            tLossSum += loss_val.item()
            if self.model_type == "Classification":
                tAccuracy += MCA(pred, labels)
            
        # Update metrics based on running loss and accuracy
        self.Metrics["Validation Loss"].append(tLossSum / len(Loader))
        if self.model_type == "Classification":
            self.Metrics["Validation Accuracy"].append(tAccuracy / len(Loader))
        
    
    # Fits model to training while also validating model 
    def fit(self, trainingLoader, validateLoader, EPOCHS):
        
        # Initate Earlystopping class to keep track of best model
        ES = EarlyStopping()
        
        for i in range(EPOCHS):
            
            # Training and Validation loop
            self.Training_Loop(trainingLoader)
            self.Validation_Loop(validateLoader)
            
            
            # Print epoch metrics
            if not self.noPrint:
                print("EPOCH:", i+1)
                print("Training Loss:", self.Metrics["Training Loss"][-1], " | Validation Loss:", self.Metrics["Validation Loss"][-1])
                if self.model_type == "Classification":
                    print("Training Accuracy:", self.Metrics["Training Accuracy"][-1].item(), " | Validation Accuracy:", self.Metrics["Validation Accuracy"][-1].item())
            
            # Check if model is overfitting and break if it is
            if ES(self.model, self.Metrics["Validation Loss"][-1]):
                if not self.noPrint:
                    print("Stopping Model Early")
                break
    
    # Evaluate model on data unseen 
    def Test_Model(self, testLoader):
        
        # Sets model into evaluation mode
        self.model.eval()
        
        # Sets up confusion matrix and accuracy
        if self.model_type == "Classification":
            confusion = ConfusionMatrix(task="multiclass", num_classes=self.classNum)
            MCA = self.accuracy(self.classNum)
        else:
            MCA = self.accuracy
        
        # A data structure for storing all labels and predictions
        predMax = torch.empty(0).to(self.device)
        labelMax = torch.empty(0).to(self.device)
        
        totalLoss = 0
        # Iterates through dataloader
        for data, labels in testLoader:

            # Moves inputs and outputs to GPU and makes the labels one-hot vectors
            data = data.to(self.device)
            if self.model_type == "Classification":
                labels = torch.eye(10)[labels]
            # else:
            #     labels = labels.reshape(-1, 1)
            labels = labels.to(self.device)

            # No Backpropagation, use no_grad to get simple prediction
            with torch.no_grad():
                pred = self.model(data)
            
            loss_val = self.Loss_Function(self.flat(pred), self.flat(labels))
            totalLoss += loss_val.item()
        
        self.Metrics["Test Loss"] = totalLoss/len(testLoader)

        
    
    # Show representations of model metrics
    def Graph_Metrics(self):
        
        # Create subplots of a certain size and spacing
        fig, (ax11, ax2) = plt.subplots(1, 2, figsize=(11,4))
        fig.subplots_adjust(wspace=0.3)
        
        # Plot loss of both training and validation on a seperate axis
        ax12 = ax11.twinx()
        ax11.plot(self.Metrics["Training Loss"], color='b')
        ax11.plot(self.Metrics["Validation Loss"], color='c')
        ax11.set_ylabel("Loss")
        ax11.legend(["Training Loss", "Validation Loss"], bbox_to_anchor=(0.40, -0.3), loc='lower right', borderaxespad=0.5)
        
        # Plot accuracy of both training and validation on a seperate axis
        if self.model_type == "Classification":
            ax12.plot(self.Metrics["Training Accuracy"], color='r')
            ax12.plot(self.Metrics["Validation Accuracy"], color='m')
            ax12.set_ylabel("Percentage")
            ax12.legend(["Training Accuracy", "Validation Accuracy"], bbox_to_anchor=(1.02, -0.3), loc='lower right', borderaxespad=0.5)

        ax11.set_title("Model Metrics Across Epochs")

        if self.model_type == "Classification":
            ax2.imshow(self.ConfMatrix, cmap='Blues')

            # Add total number of predictions for each box
            for i in range(self.ConfMatrix.shape[0]):
                for j in range(self.ConfMatrix.shape[1]):
                    ax2.text(j, i, self.ConfMatrix[i, j].item(), ha='center', va='center', color='black')

            # Removes y labels for confusion matrix
            ax2.set_xticks([])
            ax2.set_yticks([])

            ax2.set_xlabel('Predicted labels')
            ax2.set_ylabel('True labels')
            ax2.set_title("Model Confusion Matrix for Test")
        
    def reset(self):
        
        self.Metrics = {"Training Loss":[], "Validation Loss":[], 
                        "Training Accuracy":[], "Validation Accuracy":[],
                        "Test Accuracy":0} 
        
        # Place to save confidence matrix 
        self.ConfMatrix = None