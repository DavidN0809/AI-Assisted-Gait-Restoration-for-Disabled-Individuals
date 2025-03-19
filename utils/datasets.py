import os
import pandas as pd
import torch
import random
from math import gcd
from fractions import Fraction
from scipy.signal import resample_poly

class EMG_dataset(torch.utils.data.Dataset):
    def __init__(self, index_csv, lag, n_ahead, 
                 input_sensor="all", target_sensor="emg", randomize_legs=False,
                 causal_window=5):
        """
        Args:
            index_csv (str): Path to CSV listing sensor file names.
            lag (int): Number of past timesteps.
            n_ahead (int): Number of future timesteps to predict.
            input_sensor (str): Which sensor subset to use ('all', 'emg', 'acc', 'gyro').
            target_sensor (str): Which sensor to predict.
            randomize_legs (bool): If True, randomly choose which leg is input and which is target.
            causal_window (int): Window size for causal moving average smoothing.
        """
        df_index = pd.read_csv(index_csv)
        if "emg_file" not in df_index.columns:
            raise ValueError("The provided index CSV does not contain an 'emg_file' column.")
        
        self.lag = lag
        self.n_ahead = n_ahead
        self.input_sensor = input_sensor
        self.target_sensor = target_sensor
        self.randomize_legs = randomize_legs
        self.causal_window = causal_window
        self.samples = []  # list to hold (raw_data, action)
        
        # Process each CSV file listed in the index CSV.
        for _, row in df_index.iterrows():
            file_path = row["emg_file"]
            parts = [p for p in file_path.split(os.sep) if p]
            action = parts[-2] if len(parts) >= 2 else "unknown"
            
            df_sensor = pd.read_csv(file_path)
            # Drop any 'Unnamed' columns and the first column (assumed index or timestamp)
            df_sensor = df_sensor.loc[:, ~df_sensor.columns.str.contains('^Unnamed')]
            df_sensor = df_sensor.iloc[:, 1:]
            
            # --- NEW: Downsample raw data before windowing ---
            # Only downsample if the input sensor is EMG or all
            if self.input_sensor.lower() in ['emg', 'all']:
                orig_rate = 1259.259    # original high-frequency EMG rate
                target_rate = 148.148   # target lower rate (matching acc/gyro)
                ratio = Fraction(target_rate/orig_rate).limit_denominator(1000)
                up = ratio.numerator
                down = ratio.denominator
                # Now use resample_poly with these integer factors.
                data = df_sensor.to_numpy()
                # Downsample along the time axis (axis=0)
                data_downsampled = resample_poly(data, up, down, axis=0)
                # Replace the dataframe with the downsampled data (keeping the same column names)
                df_sensor = pd.DataFrame(data_downsampled, columns=df_sensor.columns)
            # ----------------------------------------------------
            
            # Apply a causal moving average filter (only current and past samples)
            df_sensor = self.causal_moving_average(df_sensor, window=self.causal_window)
            data = df_sensor.to_numpy()
            
            num_samples_file = data.shape[0] - (lag + n_ahead)
            for idx in range(num_samples_file):
                window = data[idx: idx + lag + n_ahead, :]
                self.samples.append((window, action))
                
    def __getitem__(self, idx):
        window, action = self.samples[idx]
        
        # Optionally randomize which leg is input vs. target.
        if self.randomize_legs and random.random() < 0.5:
            input_leg, target_leg = "left", "right"
        else:
            input_leg, target_leg = "right", "left"
            
        input_channels = self.get_sensor_cols(input_leg, self.input_sensor)
        target_channels = self.get_sensor_cols(target_leg, self.target_sensor)
        
        # Split the window into input (lag) and target (n_ahead) parts.
        Input_window = window[:self.lag, :]
        Target_window = window[self.lag:, :]
        
        num_columns = window.shape[1]
        if max(input_channels) >= num_columns or max(target_channels) >= num_columns:
            raise ValueError("Channel index out of bounds!")
        
        Input = Input_window[:, input_channels]
        Target = Target_window[:, target_channels]
        
        return (torch.tensor(Input, dtype=torch.float32),
                torch.tensor(Target, dtype=torch.float32),
                action)
    
    def __len__(self):
        return len(self.samples)
    
    def causal_moving_average(self, df, window=5):
        """
        Applies a causal moving average filter to each column of the dataframe.
        It uses only current and past values (non-centered) to ensure causality.
        """
        return df.rolling(window=window, min_periods=1, center=False).mean()
    
    def get_sensor_cols(self, leg, sensor_type):
        if leg.lower() == "right":
            sensors = [0, 1, 2]
        elif leg.lower() == "left":
            sensors = [3, 4, 5]
        else:
            raise ValueError("Leg must be 'left' or 'right'.")
        
        cols = []
        for s in sensors:
            base = 1 + s * 7  # Adjust based on CSV structure.
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
