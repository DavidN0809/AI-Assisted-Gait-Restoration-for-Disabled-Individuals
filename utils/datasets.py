import os
import pandas as pd
import torch
import numpy as np
import random

class EMG_dataset(torch.utils.data.Dataset):
    def __init__(self, processed_index_csv, lag, n_ahead,
                 input_sensor="all", target_sensor="emg",
                 randomize_legs=False):
        """
        If randomize_legs=True, each sample (window) chooses left or right leg
        randomly as input, and the opposite as target. For get_full_sequence(),
        we won't randomize: you must pick which leg you want to retrieve.
        """
        self.lag = lag
        self.n_ahead = n_ahead
        self.input_sensor = input_sensor.lower()
        self.target_sensor = target_sensor.lower()
        self.randomize_legs = randomize_legs

        self.num_sensors = 6
        self.samples = []  # (window_data, action)
        self.processed_columns = None
        
        # We'll store the entire content of each CSV in a list so we can return it in get_full_sequence()
        self.all_full_sequences = []  # Each item: (numpy_array, action, file_path)

        index_df = pd.read_csv(processed_index_csv)
        for _, row in index_df.iterrows():
            file_path = row["emg_file"]
            action = row.get("action", "unknown")
            if not os.path.isfile(file_path):
                print(f"Processed file not found: {file_path}. Skipping.")
                continue

            df = pd.read_csv(file_path)
            if self.processed_columns is None:
                self.processed_columns = list(df.columns)
            elif len(df.columns) != len(self.processed_columns):
                print(f"Warning: file {file_path} has {len(df.columns)} columns, "
                      f"expected {len(self.processed_columns)}. Skipping.")
                continue

            if len(df.columns) % self.num_sensors != 0:
                print(f"Warning: file {file_path} has {len(df.columns)} columns which "
                      f"is not divisible by {self.num_sensors}. Skipping.")
                continue

            data_array = df.values
            total_rows = len(data_array)

            # Save entire sequence for possible get_full_sequence usage
            self.all_full_sequences.append((data_array, action, file_path))

            # Build the small-window samples for direct training
            max_start = total_rows - (self.lag + self.n_ahead)
            if max_start < 0:
                continue

            for start_idx in range(max_start):
                window_data = data_array[start_idx : start_idx + self.lag + self.n_ahead, :]
                self.samples.append((window_data, action))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window_data, action = self.samples[idx]
        # If randomize_legs is True, randomly choose input leg and set target to the opposite
        if self.randomize_legs:
            input_leg = random.choice(["left", "right"])
            target_leg = "right" if input_leg == "left" else "left"
        else:
            # Default: use left for input, right for target
            input_leg = "left"
            target_leg = "right"

        X_array, Y_array = self._split_window(window_data, input_leg, target_leg)

        return (torch.tensor(X_array, dtype=torch.float32),
                torch.tensor(Y_array, dtype=torch.float32),
                action,
                target_leg)

    def _split_window(self, window_data, input_leg, target_leg):
        """
        Splits a window of shape [lag + n_ahead, all_columns] into:
           X -> first 'lag' rows (for the chosen input_leg's columns),
           Y -> last 'n_ahead' rows (for the chosen target_leg's columns).
        """
        # Column indices for left vs. right
        left_leg_indices = list(range(0, 21))   # 0..20
        right_leg_indices = list(range(21, 42)) # 21..41

        if input_leg == "left":
            candidate_input_cols = left_leg_indices
        else:
            candidate_input_cols = right_leg_indices

        if target_leg == "left":
            candidate_target_cols = left_leg_indices
        else:
            candidate_target_cols = right_leg_indices

        # Filter to the requested sensor type (emg/acc/gyro/all)
        if self.input_sensor != "all":
            input_cols = [
                i for i in candidate_input_cols
                if self.input_sensor in self.processed_columns[i].lower()
            ]
        else:
            input_cols = candidate_input_cols

        if self.target_sensor != "all":
            target_cols = [
                i for i in candidate_target_cols
                if self.target_sensor in self.processed_columns[i].lower()
            ]
        else:
            target_cols = candidate_target_cols

        X = window_data[:self.lag, input_cols]
        Y = window_data[self.lag:, target_cols]

        X = np.nan_to_num(X, nan=0.0)
        Y = np.nan_to_num(Y, nan=0.0)
        return X, Y

    def get_full_sequence(self, input_leg="left", target_leg="right"):
        """
        Concatenates the entire time-series (from all CSVs) into two big arrays:
            X_full: for the chosen input_leg + sensor type
            Y_full: for the chosen target_leg + sensor type

        If you only want the input side or only the target side,
        you can just use X_full or Y_full from the return.

        NOTE: This does NOT create small windows. It's the entire length
              from each file, concatenated. Transitions between files
              are simply appended end-to-end.
        """
        X_list = []
        Y_list = []

        for (data_array, action, file_path) in self.all_full_sequences:
            # We only do the column selection (no separate lag vs. n_ahead splitting).
            # The entire data_array is "one big sequence."
            # For "long-horizon" forecasts you might feed this entire X in an autoregressive loop.
            X_sub, Y_sub = self._split_entire_file(data_array, input_leg, target_leg)
            X_list.append(X_sub)
            Y_list.append(Y_sub)

        X_full = np.concatenate(X_list, axis=0) if len(X_list) > 0 else np.array([])
        Y_full = np.concatenate(Y_list, axis=0) if len(Y_list) > 0 else np.array([])

        return torch.tensor(X_full, dtype=torch.float32), torch.tensor(Y_full, dtype=torch.float32)

    def _split_entire_file(self, data_array, input_leg, target_leg):
        """
        Similar to '_split_window' but for the entire file.
        We do not separate out 'lag' or 'n_ahead' rows here.
        """
        left_leg_indices = list(range(0, 21))
        right_leg_indices = list(range(21, 42))

        if input_leg == "left":
            candidate_input_cols = left_leg_indices
        else:
            candidate_input_cols = right_leg_indices

        if target_leg == "left":
            candidate_target_cols = left_leg_indices
        else:
            candidate_target_cols = right_leg_indices

        if self.input_sensor != "all":
            input_cols = [
                i for i in candidate_input_cols
                if self.input_sensor in self.processed_columns[i].lower()
            ]
        else:
            input_cols = candidate_input_cols

        if self.target_sensor != "all":
            target_cols = [
                i for i in candidate_target_cols
                if self.target_sensor in self.processed_columns[i].lower()
            ]
        else:
            target_cols = candidate_target_cols

        X = data_array[:, input_cols]
        Y = data_array[:, target_cols]

        X = np.nan_to_num(X, nan=0.0)
        Y = np.nan_to_num(Y, nan=0.0)
        return X, Y
