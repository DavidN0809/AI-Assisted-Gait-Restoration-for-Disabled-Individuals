import os
import pandas as pd
import torch
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

class EMG_dataset(torch.utils.data.Dataset):
    """
    A dataset that returns X with shape [lag, in_channels] and Y with shape [n_ahead, out_channels].
    We do a sliding window of length (lag + n_ahead), so for each window:
      - X = the first 'lag' rows,
      - Y = the following 'n_ahead' rows.
    """
    def __init__(
        self,
        processed_index_csv,
        lag,
        n_ahead,
        input_sensor="all",
        target_sensor="all",
        sensor_pair=None,
        randomize_legs=False,
        fixed_legs=None,
        keep_time=False,
        time_col="time",
        base_dir="/data1/dnicho26/EMG_DATASET/data/final-data",
    ):
        super().__init__()
        self.lag = lag
        self.n_ahead = n_ahead
        self.input_sensor = input_sensor.lower()
        self.target_sensor = target_sensor.lower()
        self.sensor_pair = sensor_pair
        self.randomize_legs = randomize_legs
        self.fixed_legs = fixed_legs
        self.keep_time = keep_time
        self.time_col = time_col
        self.base_dir = base_dir

        self.samples = []
        self.processed_columns = None
        self.all_full_sequences = []

        index_df = pd.read_csv(processed_index_csv)

        for _, row in index_df.iterrows():
            file_path = row["file_path"]
            if file_path.startswith("./"):
                file_path = file_path[2:]
            file_path = os.path.join(self.base_dir, file_path)
            action = row.get("action", "unknown")
            if not os.path.isfile(file_path):
                print(f"File not found: {file_path}. Skipping.")
                continue

            try:
                df = pd.read_csv(file_path)
                # Drop columns with names like 'Unnamed'
                df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")],
                        inplace=True, errors="ignore")
                # Drop any rows with NaN
                df.dropna(inplace=True)
                if df.empty:
                    continue

                if self.processed_columns is None:
                    self.processed_columns = list(df.columns)

                data_array = df.values  # shape: (num_rows, num_columns)
                total_rows = len(data_array)
                # We need at least lag + n_ahead rows
                max_start = total_rows - (self.lag + self.n_ahead)
                if max_start < 0:
                    continue

                self.all_full_sequences.append((data_array, action, file_path))

                # Slide windows with stride=1
                for start_idx in range(max_start + 1):
                    window_data = data_array[start_idx : start_idx + self.lag + self.n_ahead]
                    # We'll decide input_leg / target_leg below
                    # Just store (window_data, action) for now
                    self.samples.append((window_data, action))

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        print(f"Total regression windows loaded: {len(self.samples)}")
        self.get_distribution()  # sets self.weights, etc.

    def _split_window(self, window_data, input_leg, target_leg):
        # If 'keep_time' is True and 'time_col' is present, skip the first column as time.
        offset = 1 if (self.keep_time and self.time_col in self.processed_columns) else 0

        total_sensor_cols = len(self.processed_columns) - offset
        # We assume half the columns are left, half are right
        sensors_per_leg = total_sensor_cols // 2 if total_sensor_cols >= 2 else total_sensor_cols

        left_indices = list(range(offset, offset + sensors_per_leg))
        right_indices = list(range(offset + sensors_per_leg, offset + 2 * sensors_per_leg))

        candidate_input_cols = left_indices if input_leg == "left" else right_indices
        candidate_target_cols = left_indices if target_leg == "left" else right_indices

        # Filter columns by sensor type:
        if self.input_sensor != "all":
            candidate_input_cols = [
                i for i in candidate_input_cols
                if self.input_sensor in self.processed_columns[i].lower()
            ]
        if self.target_sensor != "all":
            candidate_target_cols = [
                i for i in candidate_target_cols
                if self.target_sensor in self.processed_columns[i].lower()
            ]

        # If user specified a sensor_pair index, pick that one column from each side
        if self.sensor_pair is not None:
            if len(candidate_input_cols) > self.sensor_pair:
                candidate_input_cols = [candidate_input_cols[self.sensor_pair]]
            if len(candidate_target_cols) > self.sensor_pair:
                candidate_target_cols = [candidate_target_cols[self.sensor_pair]]

        # X is first 'lag' rows
        X = window_data[: self.lag, :][:, candidate_input_cols]
        # Y is next 'n_ahead' rows
        Y = window_data[self.lag : self.lag + self.n_ahead, :][:, candidate_target_cols]

        return X, Y

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window_data, action = self.samples[idx]
        # Decide which leg is input vs. target
        if self.fixed_legs is not None:
            input_leg, target_leg = self.fixed_legs
        elif self.randomize_legs:
            input_leg = random.choice(["left", "right"])
            target_leg = "right" if input_leg == "left" else "left"
        else:
            input_leg = "left"
            target_leg = "right"

        X_array, Y_array = self._split_window(window_data, input_leg, target_leg)

        # For weighting, we can base it on the average of the entire Y_array
        # or just the first step, etc. We'll do the entire Y.
        gt_avg = np.mean(Y_array)
        bin_value = self.value_to_bin_index(gt_avg, step=0.01, min_val=-1, max_val=1)
        weight = self.weights[bin_value]

        return (
            torch.tensor(X_array, dtype=torch.float32),  # shape: [lag, in_channels]
            torch.tensor(Y_array, dtype=torch.float32),  # shape: [n_ahead, out_channels]
            action,
            target_leg,
            weight
        )

    def get_distribution(self):
        """
        Aggregates the distribution of all values across all samples.
        Uses bins from -1 to 1 with a step of 0.01.
        """
        bins = np.arange(-1, 1.01, 0.01)
        aggregated_counts = np.zeros(len(bins) - 1, dtype=int)

        for idx in tqdm(range(len(self)), desc="Computing Distribution"):
            window_data, _ = self.samples[idx]
            # Flatten all values
            values = window_data.flatten()
            counts, _ = np.histogram(values, bins=bins)
            aggregated_counts += counts

        bin_centers = (bins[:-1] + bins[1:]) / 2
        total = np.sum(aggregated_counts)
        # Avoid /0 in empty bins
        weights = 1 / (aggregated_counts + 1e-6)
        # Scale weights to [0,1]
        weights = self.manual_min_max_scale(weights, feature_range=(0, 1))

        self.distribution = aggregated_counts
        self.weights = weights
        self.bin_centers = bin_centers

    def value_to_bin_index(self, x, step=0.01, min_val=-1, max_val=1):
        # Clip x within range
        x = max(min_val, min(x, max_val))
        num_bins = int((max_val - min_val) / step)
        index = int((x - min_val) // step)
        if index >= num_bins:
            index = num_bins - 1
        return index

    def manual_min_max_scale(self, vals, feature_range=(-1, 1)):
        min_val = np.min(vals)
        max_val = np.max(vals)
        a, b = feature_range
        if max_val == min_val:
            return np.zeros_like(vals)
        normalized = (vals - min_val) / (max_val - min_val)
        scaled = normalized * (b - a) + a
        return scaled

    def plot_distribution(self, fig_path):
        """
        Plots the aggregated distribution of EMG values.
        """
        plt.figure(figsize=(10, 6))
        bar_width = 0.005  # same as the bin step for consistency
        plt.bar(self.bin_centers, self.distribution, width=bar_width, align='center')
        plt.xlabel("Value")
        plt.ylabel("Count")
        plt.title("Aggregated Distribution of EMG Values")
        plt.savefig(fig_path)
        plt.close()
