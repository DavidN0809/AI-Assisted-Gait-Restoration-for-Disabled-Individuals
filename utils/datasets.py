import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
class EMG_dataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, lag, n_ahead):
        self.csv_file = pd.read_csv(csv_file)
        self.lag = lag
        self.n_ahead = n_ahead
        
        self.emg_files = self.csv_file.iloc[:, 0].values
        self.num_samples = self.csv_file.iloc[:, 2].values
        
        self.num_samples = self.csv_file.iloc[:, 2].values
        self.num_samples -= (lag + n_ahead)
        
        self.total_samples = sum(self.num_samples)
        self.cumulative_num_samples = np.cumsum(self.num_samples)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Find which file the index belongs to
        file_idx = np.searchsorted(self.cumulative_num_samples, idx + 1)
        if file_idx > 0:
            sample_idx = idx - self.cumulative_num_samples[file_idx - 1]
        else:
            sample_idx = idx
            
        # Load EMG data
        emg_path = self.emg_files[file_idx]
        emg_data = np.loadtxt(emg_path, delimiter=',', skiprows=1)
        emg_data = np.transpose(emg_data, (1, 0))
        Input = emg_data[:, sample_idx : sample_idx + self.lag]
        Input = np.delete(Input, np.s_[16::4], axis=0)
        Target = emg_data[16::4, sample_idx + self.lag : sample_idx + self.lag + self.n_ahead]
        
        Input = torch.permute(torch.Tensor(Input), (1, 0))
        Target = torch.permute(torch.Tensor(Target), (1, 0))
        
        return (Input, Target)
