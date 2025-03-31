import os
import pandas as pd
import torch
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt 

##############################################################################
# 2) Regression Dataset
##############################################################################
class EMG_dataset(torch.utils.data.Dataset):
    """
    A dataset for a regression task (e.g., mapping one leg's EMG to another leg's EMG).
    We create sliding windows of length (lag + n_ahead), then:
      X = first 'lag' rows (from chosen or fixed leg),
      Y = last 'n_ahead' rows (from chosen or fixed leg).
    If you supply fixed_legs = ("right", "left"), the dataset always uses right->left mapping,
    ignoring randomize_legs.
    
    Also:
      - optionally keep_time (if True, "time" column is kept as col 0).
      - filter columns by sensor type (e.g. "emg").
      - drop rows with NaN.
      - store entire sequences in all_full_sequences if you need them later.
    """
    def __init__(self, processed_index_csv, lag, n_ahead,
                 input_sensor="all", target_sensor="all",
                 randomize_legs=False,
                 fixed_legs=None,
                 keep_time=False,
                 time_col="time"):
        super().__init__()
        self.lag = lag
        self.n_ahead = n_ahead
        self.input_sensor = input_sensor.lower()
        self.target_sensor = target_sensor.lower()
        self.randomize_legs = randomize_legs
        self.fixed_legs = fixed_legs
        self.keep_time = keep_time
        self.time_col = time_col

        # We'll store (window_data, action) in self.samples.
        # window_data has shape [lag+n_ahead, selected_columns].
        self.samples = []
        self.processed_columns = None
        self.all_full_sequences = []  # For optional full-sequence usage

        # Read the CSV index (with columns: [file_path, action], etc.)
        # For testing, only use the first row:
        index_df = pd.read_csv(processed_index_csv)

        for _, row in index_df.iterrows():
            file_path = row["file_path"]
            action = row.get("action", "unknown")
            if not os.path.isfile(file_path):
                print(f"File not found: {file_path}. Skipping.")
                continue

            try:
                df = pd.read_csv(file_path)
                # Drop Unnamed columns
                df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], inplace=True, errors="ignore")
                # Drop rows with NaN
                df.dropna(inplace=True)
                if df.empty:
                    continue

                # Save processed columns if not set
                if self.processed_columns is None:
                    self.processed_columns = list(df.columns)

                data_array = df.values  # shape: (num_rows, num_columns)
                total_rows = len(data_array)
                max_start = total_rows - (self.lag + self.n_ahead)
                if max_start < 0:
                    continue

                # Keep entire sequence for get_full_sequence
                self.all_full_sequences.append((data_array, action, file_path))

                # Loop over sliding windows.
                for start_idx in range(0, max_start + 1, self.lag + self.n_ahead):
                    window_data = data_array[start_idx : start_idx + self.lag + self.n_ahead, :]
                    # Determine default target_leg: if fixed_legs is provided, use its second value;
                    # otherwise, default to "right".
                    if self.fixed_legs is not None:
                        _, target_leg = self.fixed_legs
                    else:
                        target_leg = "right"
                    # Use "left" as default input_leg.
                    _, Y_array = self._split_window(window_data, "left", target_leg)
                    # Check per ground truth signal: if for every channel the ground truth is constant, skip the window.
                    if np.all([np.all(np.diff(Y_array[:, i]) == 0) for i in range(Y_array.shape[1])]):
                        continue

                    self.samples.append((window_data, action))

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        print(f"Total regression windows loaded: {len(self.samples)}")
        self.get_distribution()

    def get_distribution(self):
        """
        Aggregates the distribution of all values across all samples.
        
        For each sample, this function computes the histogram (binned in steps of 0.01 from -1 to 1)
        and accumulates the counts into one overall distribution.
        
        It also computes the bin centers for plotting.
        
        Returns:
            tuple: (bin_centers, aggregated_counts)
        """
        bins = np.arange(-1, 1.01, 0.01)
        aggregated_counts = np.zeros(len(bins) - 1, dtype=int)
        for idx in tqdm(range(self.__len__())):
            window_data, action = self.samples[idx]
            counts, _ = np.histogram(window_data, bins=bins)
            aggregated_counts += counts
        
        bin_centers = (bins[:-1] + bins[1:]) / 2
        total = np.sum(aggregated_counts)
        weights = np.ones_like(aggregated_counts, dtype=float)
        for i, count in enumerate(aggregated_counts):
            if count > 0:
                weights[i] = (count / total)
            else:
                weights[i] = 0.0

        self.distribution = aggregated_counts
        self.weights = self.manual_min_max_scale(1 - weights, feature_range=(0, 1))
        self.bin_centers = bin_centers

    def plot_distribution(self,FIGURES_DIR):
        plt.figure(figsize=(10, 6))
        plt.bar(self.bin_centers, self.distribution, width=0.01, align='center')
        plt.xlabel("Value")
        plt.ylabel("Count")
        plt.title("Aggregated Distribution of EMG Values of the training set")
        plt.savefig(f"{FIGURES_DIR}/distribution.png")

    def value_to_bin_index(self, x, step=0.01, min_val=-1, max_val=1):
        x = max(min_val, min(x, max_val))
        num_bins = int((max_val - min_val) / step)
        index = int((x - min_val) // step)
        if index >= num_bins:
            index = num_bins - 1
        return index

    def manual_min_max_scale(self, vals, feature_range=(-1, 1)):
        min_val = np.min(vals)
        max_val = np.max(vals)
        if max_val == min_val:
            return np.zeros_like(vals)
        a, b = feature_range
        normalized = (vals - min_val) / (max_val - min_val) * (b - a) + a
        return normalized

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window_data, action = self.samples[idx]
        if self.fixed_legs is not None:
            input_leg, target_leg = self.fixed_legs
        elif self.randomize_legs:
            input_leg = random.choice(["left", "right"])
            target_leg = "right" if input_leg == "left" else "left"
        else:
            input_leg = "left"
            target_leg = "right"

        X_array, Y_array = self._split_window(window_data, input_leg, target_leg)
        gt_avg = np.mean(Y_array)
        bin_value = self.value_to_bin_index(gt_avg)
        weight = self.weights[bin_value]
        return (
            torch.tensor(X_array, dtype=torch.float32),
            torch.tensor(Y_array, dtype=torch.float32),
            action,  # can be used for debugging or evaluation
            target_leg, 
            weight
        )

    def _split_window(self, window_data, input_leg, target_leg):
        offset = 1 if (self.keep_time and self.time_col in self.processed_columns) else 0
        total_sensor_cols = len(self.processed_columns) - offset
        num_sensor_cols_per_leg = total_sensor_cols // 2

        left_start = offset
        left_end = offset + num_sensor_cols_per_leg
        right_start = left_end
        right_end = offset + 2 * num_sensor_cols_per_leg

        if input_leg == "left":
            candidate_input_cols = list(range(left_start, left_end))
        else:
            candidate_input_cols = list(range(right_start, right_end))

        if target_leg == "left":
            candidate_target_cols = list(range(left_start, left_end))
        else:
            candidate_target_cols = list(range(right_start, right_end))

        if self.input_sensor != "all":
            candidate_input_cols = [
                c for c in candidate_input_cols
                if self.input_sensor in self.processed_columns[c].lower()
            ]
        if self.target_sensor != "all":
            candidate_target_cols = [
                c for c in candidate_target_cols
                if self.target_sensor in self.processed_columns[c].lower()
            ]

        X = window_data[:self.lag, :][:, candidate_input_cols]
        Y = window_data[self.lag:, :][:, candidate_target_cols]
       
        return X, Y

    def get_full_sequence(self, input_leg="left", target_leg="right"):
        X_list, Y_list = [], []
        for (data_array, action, file_path) in self.all_full_sequences:
            X_sub, Y_sub = self._split_entire_file(data_array, input_leg, target_leg)
            X_list.append(X_sub)
            Y_list.append(Y_sub)
        if len(X_list) == 0:
            return torch.tensor([]), torch.tensor([])
        X_full = np.concatenate(X_list, axis=0)
        Y_full = np.concatenate(Y_list, axis=0)
        return torch.tensor(X_full, dtype=torch.float32), torch.tensor(Y_full, dtype=torch.float32)

    def _split_entire_file(self, data_array, input_leg, target_leg):
        offset = 1 if (self.keep_time and self.time_col in self.processed_columns) else 0
        left_start = offset
        left_end = offset + 21
        right_start = offset + 21
        right_end = offset + 42

        if input_leg == "left":
            candidate_input_cols = list(range(left_start, left_end))
        else:
            candidate_input_cols = list(range(right_start, right_end))

        if target_leg == "left":
            candidate_target_cols = list(range(left_start, left_end))
        else:
            candidate_target_cols = list(range(right_start, right_end))

        if self.input_sensor != "all":
            candidate_input_cols = [
                c for c in candidate_input_cols
                if self.input_sensor in self.processed_columns[c].lower()
            ]
        if self.target_sensor != "all":
            candidate_target_cols = [
                c for c in candidate_target_cols
                if self.target_sensor in self.processed_columns[c].lower()
            ]

        X = data_array[:, candidate_input_cols]
        Y = data_array[:, candidate_target_cols]
        return X, Y
