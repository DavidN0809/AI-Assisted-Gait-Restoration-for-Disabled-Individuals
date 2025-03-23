import os
import pandas as pd
import torch
import numpy as np
import random

class EMG_dataset(torch.utils.data.Dataset):
    def __init__(self, processed_index_csv, lag, n_ahead, input_sensor="all", target_sensor="emg", randomize_legs=False):
        """
        Initializes the dataset by reading the processed CSV files.
        The new index CSV is expected to have at least:
          - 'processed_file': path to the processed CSV
          - 'action': the action label (and potentially other metadata such as uuid)
        This version includes a check that all processed CSV files have a consistent number
        of columns and that the total number of columns is divisible by 6 (i.e., 6 sensors).
        """
        self.lag = lag
        self.n_ahead = n_ahead
        self.input_sensor = input_sensor.lower()
        self.target_sensor = target_sensor.lower()
        self.randomize_legs = randomize_legs

        self.num_sensors = 6
        self.samples = []  # List to store (window_data, action)
        self.processed_columns = None  # To store columns from first valid file

        # Load the new index CSV
        index_df = pd.read_csv(processed_index_csv)
        for _, row in index_df.iterrows():
            file_path = row["emg_file"]
            action = row.get("action", "unknown")
            if not os.path.isfile(file_path):
                print(f"Processed file not found: {file_path}. Skipping.")
                continue
            df = pd.read_csv(file_path)
            
            # Check that the file has a consistent number of columns.
            if self.processed_columns is None:
                self.processed_columns = list(df.columns)
                #print("Processed columns:", self.processed_columns)
            elif len(df.columns) != len(self.processed_columns):
                print(f"Warning: file {file_path} has {len(df.columns)} columns, expected {len(self.processed_columns)}. Skipping.")
                continue

            # Check that the number of columns is divisible by the number of sensors.
            if len(df.columns) % self.num_sensors != 0:
                print(f"Warning: file {file_path} has {len(df.columns)} columns which is not divisible by {self.num_sensors} sensors. Skipping.")
                continue

            data_array = df.values
            total_rows = len(data_array)
            max_start = total_rows - (lag + n_ahead)
            if max_start < 0:
                continue
            for start_idx in range(max_start):
                window_data = data_array[start_idx : start_idx + lag + n_ahead, :]
                self.samples.append((window_data, action))
                    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        window_data, action = self.samples[idx]
        # window_data shape: [lag + n_ahead, total_features] where total_features=42
        # Choose legs based on the randomize_legs flag:
        if self.randomize_legs:
            # Randomly pick one leg for the input and use the opposite for ground truth.
            input_leg = random.choice(["left", "right"])
            target_leg = "right" if input_leg == "left" else "left"
        else:
            # Default: use the left leg for both input and target.
            input_leg = "left"
            target_leg = "left"
        
        # Determine the column indices for left and right legs.
        # (Assuming left leg is columns 0–20 and right leg is columns 21–41)
        left_leg_indices = list(range(0, 21))
        right_leg_indices = list(range(21, 42))
        
        # For the input window, select indices for the chosen leg based on input_sensor.
        if input_leg == "left":
            candidate_input_cols = left_leg_indices
        else:
            candidate_input_cols = right_leg_indices
        
        if self.input_sensor.lower() != "all":
            # Filter the candidate indices using the modality string contained in the column names.
            input_cols = [i for i in candidate_input_cols 
                        if self.input_sensor.lower() in self.processed_columns[i].lower()]
        else:
            input_cols = candidate_input_cols
        
        # For the target window, select indices for the target leg based on target_sensor.
        if target_leg == "left":
            candidate_target_cols = left_leg_indices
        else:
            candidate_target_cols = right_leg_indices
        
        if self.target_sensor.lower() != "all":
            target_cols = [i for i in candidate_target_cols 
                        if self.target_sensor.lower() in self.processed_columns[i].lower()]
        else:
            target_cols = candidate_target_cols
        
        # Extract the time windows:
        # X is taken from the first self.lag timesteps (using the chosen input columns)
        # Y is taken from the next self.n_ahead timesteps (using the chosen target columns)
        X_array = window_data[:self.lag, :][:, input_cols]
        Y_array = window_data[self.lag:, :][:, target_cols]
        
        # Replace any NaN values with 0.0.
        X_array = np.nan_to_num(X_array, nan=0.0)
        Y_array = np.nan_to_num(Y_array, nan=0.0)
        
        # Return a tuple that includes:
        #   X: input window from the chosen leg and modality,
        #   Y: target window from the ground-truth leg and modality,
        #   action: the action label,
        #   target_leg: a string indicating which leg the ground truth comes from.
        return (torch.tensor(X_array, dtype=torch.float32),
                torch.tensor(Y_array, dtype=torch.float32),
                action,
                target_leg)

