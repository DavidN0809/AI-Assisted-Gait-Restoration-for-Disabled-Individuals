import os
os.chdir('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from ptflops import get_model_complexity_info

from torch.utils.data import random_split
from torch.utils.data import TensorDataset, DataLoader
from utils.Trainer import ModelTrainer
from models.models import BasicLSTM

device = "cuda" if torch.cuda.is_available() else "cpu"
from utils.datasets import EMG_dataset


# skeleton_path = "M:/Datasets/shock_walk/Videos/skeletons/"
# emg_path = "M:/Datasets/shock_walk/processed/"

# skeleton_files = [skeleton_path + f for f in os.listdir(skeleton_path)]
# emg_files = [emg_path + f for f in os.listdir(emg_path)]

# summary_csv = pd.DataFrame({"EMG PATH" : emg_files, "SKEL PATH": skeleton_files, "FRAMES": [1800] * len(skeleton_files)})
# summary_csv.to_csv("emg_skel.csv", index=False)

def train_test_model(model, lr, trainLoader, validateLoader, testLoader, EPOCHS, noPrint=False):
    Trainer = ModelTrainer(model, nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=lr), nn.MSELoss(), "Regression",  device, 0, noPrint=noPrint)
    t0 = datetime.now()
    Trainer.fit(trainLoader, validateLoader, EPOCHS)
    t1 = datetime.now()
    Trainer.Test_Model(testLoader)
    
    if not noPrint:
        print("\nTest Loss:", Trainer.Metrics["Test Loss"], "\nTime to Train:", t1 - t0)
        Trainer.Graph_Metrics()
    
    # macs, params = get_model_complexity_info(Trainer.model, (24, 4), as_strings=True, print_per_layer_stat=False, verbose=True)
    Trainer.model.to("cpu")
    
    return Trainer, None # (macs, params)

dataset = EMG_dataset(r"D:\UNC Charlotte Dropbox\orgs-ecgr-QuantitativeImagingandAILaboratory\preprocessed\index.csv", lag=60, n_ahead=12)
X, Y = dataset.__getitem__(0)
print(X.shape, Y)

dataset = EMG_dataset("emg_skel.csv", lag=30, n_ahead=10)

batch_size = 12

train_dataset, valid_dataset, test_dataset = random_split(dataset, [0.7, 0.1, 0.2])
testLoader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=True)


for X, Y in testLoader:
    print(X.shape, Y.shape)
    
#     y_range = range(len(X[0][:, 0]), len(X[0][:, 0]) + len(Y[0][:, 0]))
#     # First Batch
#     plt.plot(X[0][:, 0])
#     plt.plot(y_range, Y[0][:, 0])
#     break

input_size = 28
hidden_size = 128
num_layers = 5
output_size = train_dataset.dataset.n_ahead

trainLoader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=False)
validLoader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=4, shuffle=False)


model = BasicLSTM(input_size, hidden_size, num_layers, output_size)

Trainer, mp = train_test_model(model, 0.0007, trainLoader, validLoader, testLoader, EPOCHS=300, noPrint=False)
print(mp)

# model_path = "./models/pre-trained/EMG_32.pt"
# model = BasicLSTM(input_size, hidden_size, num_layers, output_size)
# model.load_state_dict(torch.load(model_path))

for X, Y in testLoader:
    X = X
    Y = Y
    pred = model(X).detach().numpy()
    
    print(pred[0,:,0].shape, Y.shape, X[0].shape)
    
    # Assume Extension 
    y_range = range(len(X[0]), len(X[0])+validLoader.dataset.dataset.n_ahead)

    
    # plt.plot(X[0,:].detach().cpu())
    plt.plot(y_range, pred[0,:], 'b')
    plt.plot(y_range, Y[0,:], 'g')
    
    # Create Line2D objects representing the lines
    line_pred = plt.Line2D([], [], color='blue')
    line_Y = plt.Line2D([], [], color='green')

    # Create the legend using the Line2D objects and labels
    plt.legend([line_pred, line_Y], ["Prediction", "Actual"])
    
    break

torch.save(Trainer.model.state_dict(), "models/EMG_32_Full2.pt")
