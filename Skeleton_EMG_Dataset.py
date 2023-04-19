import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class SkeletonEMGDataset(Dataset):
    def __init__(self, csv_file, window_size=10, resolution=(1920, 600)):
        self.csv_file = pd.read_csv(csv_file)
        self.window_size = window_size
        self.resolution = resolution
        
        self.emg_files = self.csv_file.iloc[:, 0].values
        self.skeleton_files = self.csv_file.iloc[:, 1].values
        self.num_samples = self.csv_file.iloc[:, 2].values
        self.num_samples -= window_size + 1
        
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
        emg_data = np.loadtxt(emg_path, delimiter=',', skiprows=1, usecols=range(8))
        emg_window = emg_data[sample_idx + self.window_size + 1]

        # Load skeleton data
        skeleton_path = self.skeleton_files[file_idx]
        skeleton_data = np.load(skeleton_path)
        skeleton_window = skeleton_data[sample_idx:sample_idx + self.window_size]

        skeleton_window[:, :, 0] /= self.resolution[0]
        skeleton_window[:, :, 1] /= self.resolution[1]

        for i in range(3):
            for j in range(skeleton_window.shape[1]):
                skeleton_window[:, j, i] -= skeleton_window[:, 0, i]

        return torch.from_numpy(skeleton_window).float(), torch.from_numpy(emg_window).float()