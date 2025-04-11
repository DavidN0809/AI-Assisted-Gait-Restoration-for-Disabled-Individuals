import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
from scipy import signal
from scipy.fft import fft, fftfreq
import pywt

class EMG_dataset(torch.utils.data.Dataset):
    """
    Dataset class that assumes a CSV with 42 columns: 
      - columns 0-20 (first half) are left-leg sensors (inputs)
      - columns 21-41 (second half) are right-leg sensors (targets)
    
    When a sensor substring is provided (e.g. "emg"), only the columns whose names
    contain the substring (in the proper half) are selected.
    
    A sliding window (of length lag+n_ahead) with configurable overlap is used.
    """
    def __init__(
        self,
        processed_index_csv,
        lag,
        n_ahead,
        overlap=0.0,
        input_sensor="all",
        target_sensor="all",
        base_dir="/data1/dnicho26/EMG_DATASET/data/final-data",
    ):
        super().__init__()
        self.lag = lag
        self.n_ahead = n_ahead
        self.overlap = overlap
        self.input_sensor = input_sensor.lower()
        self.target_sensor = target_sensor.lower()
        self.base_dir = base_dir

        self.samples = []
        self.processed_columns = None
        self.all_full_sequences = []

        index_df = pd.read_csv(processed_index_csv)
        window_length = self.lag + self.n_ahead

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
                # Drop any "Unnamed" columns and rows with NaN values.
                df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")],
                        inplace=True, errors="ignore")
                df.dropna(inplace=True)
                if df.empty:
                    continue

                if self.processed_columns is None:
                    self.processed_columns = list(df.columns)
                data_array = df.values
                total_rows = len(data_array)
                max_start = total_rows - window_length
                if max_start < 0:
                    continue

                self.all_full_sequences.append((data_array, action, file_path))
                # Compute stride using overlap (e.g., 0.5 => 50% overlap)
                stride = max(1, int(window_length * (1 - self.overlap)))
                for start_idx in range(0, max_start + 1, stride):
                    window_data = data_array[start_idx : start_idx + window_length]
                    self.samples.append((window_data, action))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        print(f"Total regression windows loaded: {len(self.samples)}")
        self.get_distribution()

    def _split_window(self, window_data):
        """
        Splits window_data into input (first 'lag' rows) and target (next 'n_ahead' rows),
        using a helper to filter sensor columns.

        The left half (columns 0-20) is used for input and the right half (columns 21-41) for target.
        """
        total_cols = len(self.processed_columns)
        half = total_cols // 2  # Expecting first half and second half split

        def filter_columns(lower, upper, sensor):
            """Return indices between lower and upper that contain the sensor substring or all if 'all'."""
            if sensor == "all":
                return list(range(lower, upper))
            # Filter based on if the column name contains the sensor substring.
            return [i for i in range(lower, upper) if sensor in self.processed_columns[i].lower()]

        input_cols = filter_columns(0, half, self.input_sensor)
        target_cols = filter_columns(half, total_cols, self.target_sensor)

        # Extract X and Y windows
        X = window_data[: self.lag, :][:, input_cols]
        Y = window_data[self.lag : self.lag + self.n_ahead, :][:, target_cols]
        return X, Y

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window_data, action = self.samples[idx]
        X_array, Y_array = self._split_window(window_data)
        
        # Compute a weight based on the mean of the target values.
        gt_avg = np.mean(Y_array)
        bin_value = self.value_to_bin_index(gt_avg, step=0.01, min_val=0, max_val=1)
        weight = self.weights[bin_value]
        
        return (
            torch.tensor(X_array, dtype=torch.float32),  # shape: [lag, in_channels]
            torch.tensor(Y_array, dtype=torch.float32),  # shape: [n_ahead, out_channels]
            action,
            weight,
        )

    def get_distribution(self):
        """
        Aggregates the distribution of all sensor values across samples using bins from 0 to 1 (step 0.01).
        """
        bins = np.arange(0, 1.01, 0.01)
        aggregated_counts = np.zeros(len(bins) - 1, dtype=int)
        for idx in tqdm(range(len(self)), desc="Computing Distribution"):
            window_data, _ = self.samples[idx]
            values = window_data.flatten()
            counts, _ = np.histogram(values, bins=bins)
            aggregated_counts += counts

        bin_centers = (bins[:-1] + bins[1:]) / 2
        weights = 1 / (aggregated_counts + 1e-6)  # avoid division by zero
        weights = self.manual_min_max_scale(weights, feature_range=(0, 1))

        self.distribution = aggregated_counts
        self.weights = weights
        self.bin_centers = bin_centers

    def value_to_bin_index(self, x, step=0.01, min_val=0, max_val=1):
        x = max(min_val, min(x, max_val))
        num_bins = int((max_val - min_val) / step)
        index = int((x - min_val) // step)
        if index >= num_bins:
            index = num_bins - 1
        return index

    def manual_min_max_scale(self, vals, feature_range=(-1, 1)):
        a, b = feature_range
        min_val = np.min(vals)
        max_val = np.max(vals)
        if max_val == min_val:
            return np.zeros_like(vals)
        normalized = (vals - min_val) / (max_val - min_val)
        return normalized * (b - a) + a

    def plot_distribution(self, fig_path):
        """
        Plots the aggregated distribution of sensor values.
        """
        plt.figure(figsize=(10, 6))
        bar_width = 0.005  # matches the bin step
        plt.bar(self.bin_centers, self.distribution, width=bar_width, align='center')
        plt.xlabel("Value")
        plt.ylabel("Count")
        plt.title("Aggregated Distribution of Sensor Values")
        plt.savefig(fig_path)
        plt.close()


class EMG_dataset_with_features(EMG_dataset):
    """
    Dataset class that extends EMG_dataset and includes feature extraction.
    Uses time and frequency domain features for each input channel.
    """
    
    def _extract_time_features(self, signal_window):
        """Extract time-domain features from a signal window"""
        features = {}
        
        # Mean Absolute Value (MAV)
        features['mav'] = np.mean(np.abs(signal_window))
        
        # Waveform Length (WL)
        features['wl'] = np.sum(np.abs(np.diff(signal_window)))
        
        # Root Mean Square (RMS)
        features['rms'] = np.sqrt(np.mean(signal_window**2))
        
        # Zero Crossings (ZC)
        features['zc'] = ((signal_window[:-1] * signal_window[1:]) < 0).sum()
        
        # Difference RMS (DRMS)
        diff_signal = np.diff(signal_window)
        features['drms'] = np.sqrt(np.mean(diff_signal**2))
        
        return features
    
    def _extract_freq_features(self, signal_window, fs=1000):
        """Extract frequency-domain features from a signal window"""
        features = {}
        
        # Compute FFT
        n = len(signal_window)
        yf = fft(signal_window)
        xf = fftfreq(n, 1/fs)[:n//2]
        
        # STFT features (averaged over three frequency bands)
        f, t, Zxx = signal.stft(signal_window, fs=fs, nperseg=min(64, len(signal_window)))
        
        # Define frequency bands
        bands = [(0, 50), (50, 150), (150, 500)]
        for i, (low, high) in enumerate(bands):
            band_mask = (f >= low) & (f <= high)
            if np.any(band_mask):
                features[f'stft_band{i}_mean'] = np.mean(np.abs(Zxx[band_mask]))
                features[f'stft_band{i}_std'] = np.std(np.abs(Zxx[band_mask]))
        
        # SWT feature (using level-3 detail coefficients)
        max_level = pywt.swt_max_level(len(signal_window))
        safe_level = min(3, max_level)
        coeffs = pywt.swt(signal_window, 'db1', level=safe_level)
        if safe_level >= 3:
            detail_coeffs = coeffs[2][1]
        else:
            detail_coeffs = coeffs[-1][1]
        features['swt_level3'] = np.mean(np.abs(detail_coeffs))
        
        return features
    
    def _extract_features(self, signal_window, fs=1000):
        """Combine time and frequency domain features"""
        time_features = self._extract_time_features(signal_window)
        freq_features = self._extract_freq_features(signal_window, fs)
        return {**time_features, **freq_features}
    
    def __getitem__(self, idx):
        window_data, action = self.samples[idx]
        X_array, Y_array = self._split_window(window_data)
        
        # Extract features for each channel in the input window
        feature_dicts = []
        for channel in range(X_array.shape[1]):
            channel_signal = X_array[:, channel]
            features = self._extract_features(channel_signal)
            feature_dicts.append(features)
        
        # Convert feature dicts to numpy array
        feature_names = sorted(feature_dicts[0].keys()) if feature_dicts else []
        features_array = np.zeros((len(feature_dicts), len(feature_names)))
        
        for i, fd in enumerate(feature_dicts):
            features_array[i] = [fd[name] for name in feature_names]
        
        # Compute a weight based on the mean of the target values.
        gt_avg = np.mean(Y_array)
        bin_value = self.value_to_bin_index(gt_avg, step=0.01, min_val=0, max_val=1)
        weight = self.weights[bin_value]
        
        return (
            torch.tensor(X_array, dtype=torch.float32),  # shape: [lag, in_channels]
            torch.tensor(Y_array, dtype=torch.float32),  # shape: [n_ahead, out_channels]
            torch.tensor(features_array, dtype=torch.float32),  # shape: [in_channels, num_features]
            action,
            weight,
        )


class EMG_dataset_window_norm(EMG_dataset):
    """
    Dataset class that extends EMG_dataset and normalizes each window individually.
    Each window is normalized by column (per channel) for both input and target data.
    """
    
    def __init__(self, processed_index_csv, lag, n_ahead, input_sensor, target_sensor, base_dir, input_size=None, trail_dir=None):
        self.input_size = input_size
        if trail_dir is not None:
            base_dir = trail_dir
        super().__init__(processed_index_csv, lag, n_ahead, input_sensor=input_sensor, target_sensor=target_sensor, base_dir=base_dir)

    def __getitem__(self, idx):
        window_data, action = self.samples[idx]
        X_array, Y_array = self._split_window(window_data)
        
        # Normalize X_array (input) per column within the window to range [-1, 1]
        X_min = np.min(X_array, axis=0, keepdims=True)
        X_max = np.max(X_array, axis=0, keepdims=True)
        X_range = X_max - X_min + 1e-8  # Add epsilon to avoid division by zero
        X_array_norm = 2 * ((X_array - X_min) / X_range) - 1
        
        # Normalize Y_array (target) per column within the window to range [-1, 1]
        Y_min = np.min(Y_array, axis=0, keepdims=True)
        Y_max = np.max(Y_array, axis=0, keepdims=True)
        Y_range = Y_max - Y_min + 1e-8
        Y_array_norm = 2 * ((Y_array - Y_min) / Y_range) - 1
        
        # Compute a weight based on the mean of the target values (using original values)
        gt_avg = np.mean(Y_array)
        bin_value = self.value_to_bin_index(gt_avg, step=0.01, min_val=0, max_val=1)
        weight = self.weights[bin_value]
        
        return (
            torch.tensor(X_array_norm, dtype=torch.float32),  # shape: [lag, in_channels]
            torch.tensor(Y_array_norm, dtype=torch.float32),  # shape: [n_ahead, out_channels]
            action,
            weight,
        )