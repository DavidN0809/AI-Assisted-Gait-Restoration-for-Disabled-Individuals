import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class EMG_dataset(torch.utils.data.Dataset):
    def __init__(self, index_csv, lag, n_ahead, 
                 input_sensor="all", target_sensor="emg"):
        """
        Args:
            index_csv (str): Path to the index CSV file which contains the sensor data file path.
            lag (int): Number of past timesteps as input.
            n_ahead (int): Number of future timesteps as target.
            input_leg (str): 'right' or 'left' – selects which leg's sensor data is used as input.
            target_leg (str): 'right' or 'left' – selects which leg's sensor data is used as target.
            input_sensor (str): One of "all", "emg", "acc", "gyro" – which sensor types to use for input.
            target_sensor (str): One of "all", "emg", "acc", "gyro" – which sensor types to use for target.
        """
        # Load the index CSV.
        df_index = pd.read_csv(index_csv)
        
        # Ensure the CSV contains the expected column.
        if "emg_file" not in df_index.columns:
            raise ValueError("The provided index CSV does not contain an 'emg_file' column.")
        
        sensor_data_path = df_index["emg_file"].iloc[0]
        df_sensor = pd.read_csv(sensor_data_path)
        print("DataFrame shape:", df_sensor.shape)
        print("Columns:", df_sensor.columns.tolist())


        # Drop the first column (assumed to be an index) if necessary.
        df_sensor = df_sensor.iloc[:, 1:]
        
        # Convert the sensor dataframe to a numpy array.
        self.data = df_sensor.to_numpy()
        self.lag = lag
        self.n_ahead = n_ahead
        self.num_samples = self.data.shape[0] - (lag + n_ahead)
        
        # Define sensor selection:
        # The first column is assumed to be "time". Then, sensor blocks follow.
        # Each sensor block consists of 7 columns in the following order:
        #   [EMG, ACC X, ACC Y, ACC Z, GYRO X, GYRO Y, GYRO Z]
        # Sensors 0-2 (i.e. first 3 blocks) correspond to the right leg,
        # Sensors 3-5 (i.e. blocks 4-6) correspond to the left leg.
        def get_sensor_cols(leg, sensor_type):
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
            print(f"For leg {leg} and sensor {sensor_type}, computed columns: {cols}")
            return cols

        self.input_channels = get_sensor_cols("right", input_sensor)
        self.target_channels = get_sensor_cols("left", target_sensor)
    

    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        Input_window = self.data[idx: idx + self.lag, :]
        Target_window = self.data[idx + self.lag: idx + self.lag + self.n_ahead, :]

        # Debug: Check max indices
        max_input_index = max(self.input_channels)
        max_target_index = max(self.target_channels)
        num_columns = self.data.shape[1]
        #print("Max input index:", max_input_index, "Max target index:", max_target_index, "Number of columns:", num_columns)
        
        assert max_input_index < num_columns, "Input channel index out of bounds!"
        assert max_target_index < num_columns, "Target channel index out of bounds!"

        Input = Input_window[:, self.input_channels]
        Target = Target_window[:, self.target_channels]

        return torch.tensor(Input, dtype=torch.float32), torch.tensor(Target, dtype=torch.float32)
