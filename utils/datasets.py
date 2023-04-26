import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class SkeletonEMGDataset(Dataset):
    def __init__(self, csv_file, n_ahead=1, window_size=10, resolution=(1920, 1920)):
        self.csv_file = pd.read_csv(csv_file)
        self.window_size = window_size
        self.resolution = resolution
        print(resolution)
        self.n_ahead = n_ahead
        
        self.emg_files = self.csv_file.iloc[:, 0].values
        self.skeleton_files = self.csv_file.iloc[:, 1].values
        self.num_samples = self.csv_file.iloc[:, 2].values
        self.num_samples -= window_size + n_ahead
        
        self.total_samples = sum(self.num_samples)
        self.cumulative_num_samples = np.cumsum(self.num_samples)
        
    def __len__(self):
        return self.cumulative_num_samples[-1]
    
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
        
        emg_window = emg_data[0::4, sample_idx + self.window_size : sample_idx + self.window_size + self.n_ahead]

        # Load skeleton data
        skeleton_path = self.skeleton_files[file_idx]
        skeleton_data = np.load(skeleton_path, allow_pickle=True)
        
        skeleton_window = skeleton_data[sample_idx : sample_idx + self.window_size]
        skeleton_window[:, :, 0] /= self.resolution[0]
        skeleton_window[:, :, 1] /= self.resolution[1]

        for i in range(3):
            for j in range(skeleton_window.shape[1]):
                skeleton_window[:, j, i] -= skeleton_window[:, 0, i]

        return torch.from_numpy(skeleton_window).float(), torch.from_numpy(emg_window).float()
    
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

class EMGO_dataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, lag, n_ahead):
        self.csv_file = pd.read_csv(csv_file)
        display(self.csv_file)
        self.lag = lag
        self.n_ahead = n_ahead
        
        self.emg_files = self.csv_file.iloc[:, 0].values
        self.num_samples = self.csv_file.iloc[:, 1].values
        
        self.num_samples = self.csv_file.iloc[:, 1].values
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
        Input = emg_data[0::4, sample_idx : sample_idx + self.lag][:4]
        
        Target = emg_data[16::4, sample_idx + self.lag : sample_idx + self.lag + self.n_ahead]
        
        Input = torch.permute(torch.Tensor(Input), (1, 0))
        Target = torch.permute(torch.Tensor(Target), (1, 0))
        
        return (Input, Target)