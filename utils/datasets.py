# datasets.py
import os
import pandas as pd
import torch
import random

class EMG_dataset(torch.utils.data.Dataset):
    def __init__(self, index_csv, lag, n_ahead, 
                 input_sensor="all", target_sensor="emg", randomize_legs=False):
        """
        Args:
            randomize_legs (bool): If True, randomly choose which leg is input and which is target on each fetch.
        """
        df_index = pd.read_csv(index_csv)
        if "emg_file" not in df_index.columns:
            raise ValueError("The provided index CSV does not contain an 'emg_file' column.")
        
        self.lag = lag
        self.n_ahead = n_ahead
        self.input_sensor = input_sensor
        self.target_sensor = target_sensor
        self.randomize_legs = randomize_legs
        self.samples = []  # List to hold (raw_data, action)
        
        # Process each file listed in the index CSV.
        for _, row in df_index.iterrows():
            file_path = row["emg_file"]
            parts = [p for p in file_path.split(os.sep) if p]  # remove empty strings
            action = parts[-2] if len(parts) >= 2 else "unknown"
            
            # Read the sensor CSV file.
            df_sensor = pd.read_csv(file_path)
            df_sensor = df_sensor.iloc[:, 1:]  # Drop the first column (assumed index)
            data = df_sensor.to_numpy()
            
            # Instead of computing fixed Input and Target windows, store the raw data with starting indices.
            num_samples_file = data.shape[0] - (lag + n_ahead)
            for idx in range(num_samples_file):
                # Store the complete sliding window (lag + n_ahead) for later slicing.
                window = data[idx: idx + lag + n_ahead, :]
                self.samples.append((window, action))
                
    def __getitem__(self, idx):
        window, action = self.samples[idx]
        
        # Randomly choose which leg is input and which is target if enabled.
        if self.randomize_legs and random.random() < 0.5:
            input_leg, target_leg = "left", "right"
        else:
            input_leg, target_leg = "right", "left"
            
        # Compute the channel indices using the provided helper.
        input_channels = self.get_sensor_cols(input_leg, self.input_sensor)
        target_channels = self.get_sensor_cols(target_leg, self.target_sensor)
        
        # Slice the window: first part is for input, second part for target.
        Input_window = window[:self.lag, :]
        Target_window = window[self.lag:, :]
        
        # Verify channel indices are within bounds.
        max_input_index = max(input_channels)
        max_target_index = max(target_channels)
        num_columns = window.shape[1]
        if max_input_index >= num_columns or max_target_index >= num_columns:
            raise ValueError("Channel index out of bounds!")
        
        Input = Input_window[:, input_channels]
        Target = Target_window[:, target_channels]
        
        return (torch.tensor(Input, dtype=torch.float32),
                torch.tensor(Target, dtype=torch.float32),
                action)
    
    def __len__(self):
        return len(self.samples)
    
    # Include the helper function inside the class (or import it if defined elsewhere)
    def get_sensor_cols(self, leg, sensor_type):
        if leg.lower() == "right":
            sensors = [0, 1, 2]
        elif leg.lower() == "left":
            sensors = [3, 4, 5]
        else:
            raise ValueError("Leg must be 'left' or 'right'.")
        
        cols = []
        for s in sensors:
            base = 1 + s * 7  # Adjust according to your CSV structure
            if sensor_type.lower() == "emg":
                cols.append(base)
            elif sensor_type.lower() == "acc":
                cols.extend([base + 1, base + 2, base + 3])
            elif sensor_type.lower() == "gyro":
                cols.extend([base + 4, base + 5, base + 6])
            elif sensor_type.lower() == "all":
                cols.extend(list(range(base, base + 7)))
            else:
                raise ValueError("Sensor type must be one of 'all', 'emg', 'acc', or 'gyro'.")
        return cols
