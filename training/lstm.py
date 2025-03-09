import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

from torch.utils.data import random_split, DataLoader
from utils.Trainer import ModelTrainer
from models.models import BasicLSTM

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
from utils.datasets import EMG_dataset

import logging
logging.disable(logging.CRITICAL)

def train_test_model(model, lr, trainLoader, validateLoader, testLoader, EPOCHS, noPrint=False):
    Trainer = ModelTrainer(
        model, 
        nn.MSELoss(), 
        torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5),
        nn.MSELoss(), 
        "Regression",  
        device, 
        0, 
        noPrint=noPrint
    )
    t0 = datetime.now()
    Trainer.fit(trainLoader, validateLoader, EPOCHS)
    t1 = datetime.now()
    Trainer.Test_Model(testLoader)
    
    if not noPrint:
        print("\nTest Loss:", Trainer.Metrics["Test Loss"], "\nTime to Train:", t1 - t0)
        Trainer.Graph_Metrics()
    
    # Move the model to CPU before returning.
    Trainer.model.to("cpu")
    
    return Trainer, None

if __name__ == '__main__':

#    file_path = r"D:\UNC Charlotte Dropbox\orgs-ecgr-QuantitativeImagingandAILaboratory\preprocessed\index.csv"
    file_path = "/data1/dnicho26/EMG_DATASET/data/preprocessed/index.csv"
    # Split dataset into training, validation, and test sets.
    batch_size = 12
    lag=30
    n_ahead=10

    # Determine the input size.
    # For example, if each sensor produces 7 features (e.g., EMG, ACC X, ACC Y, ACC Z, GYRO X, GYRO Y, GYRO Z)
    # then input_size = 3 * 7 = 21.
    input_size = 21  # Adjust based on your sensor's actual feature count.
    hidden_size = 128
    num_layers = 5
    #output_size = dataset.n_ahead  # Typically, the forecast horizon.
    output_size = 21
    epochs=300
    lr=0.00007

    # Create the dataset. Update the path to your index CSV as needed.
    dataset = EMG_dataset(
        file_path, 
        lag=lag, 
        n_ahead=n_ahead, 
        input_leg="left"
    )
    
    # Check an example window
    X, Y = dataset.__getitem__(0)
    print("Example input and target shapes:", X.shape, Y.shape)

    train_dataset, valid_dataset, test_dataset = random_split(dataset, [0.7, 0.1, 0.2])

    testLoader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    trainLoader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    validLoader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=0, shuffle=False)

    model = BasicLSTM(input_size, hidden_size, num_layers, output_size, n_ahead)
    
    Trainer, mp = train_test_model(model, lr, trainLoader, validLoader, testLoader, EPOCHS=epochs, noPrint=False)
    torch.save(Trainer.model.state_dict(), "models/EMG_32_Full2.pt")
