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
import datetime

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
        keep_time=False,
        single_file_mode=False,
    ):
        super().__init__()
        self.lag = lag
        self.n_ahead = n_ahead
        self.overlap = overlap
        self.input_sensor = input_sensor.lower()
        self.target_sensor = target_sensor.lower()
        self.base_dir = base_dir
        self.keep_time = keep_time
        self.single_file_mode = single_file_mode

        self.samples = []
        self.processed_columns = None
        self.time_columns = None
        self.all_full_sequences = []

        # Handle single file mode vs index CSV mode
        if self.single_file_mode:
            # In single file mode, processed_index_csv is the direct path to the CSV file
            file_path = processed_index_csv
            self._process_file(file_path, "unknown")
        else:
            # Normal mode using an index CSV
            index_df = pd.read_csv(processed_index_csv)
            for _, row in index_df.iterrows():
                file_path = row["file_path"]
                if file_path.startswith("./"):
                    file_path = file_path[2:]
                file_path = os.path.join(self.base_dir, file_path)
                action = row.get("action", "unknown")
                self._process_file(file_path, action)

        print(f"Total regression windows loaded: {len(self.samples)}")
        self.get_distribution()
        
    def _process_file(self, file_path, action):
        """Process a single CSV file and add its windows to the samples list"""
        window_length = self.lag + self.n_ahead
        
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}. Skipping.")
            return
            
        try:
            df = pd.read_csv(file_path)
            # Drop any "Unnamed" columns and rows with NaN values.
            df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")],
                    inplace=True, errors="ignore")
            df.dropna(inplace=True)
            if df.empty:
                return

            # Identify time columns
            time_columns = [col for col in df.columns if 'time' in col.lower()]
            non_time_columns = [col for col in df.columns if 'time' not in col.lower()]
            
            if self.processed_columns is None:
                if self.keep_time:
                    self.processed_columns = list(df.columns)
                    self.time_columns = time_columns
                else:
                    self.processed_columns = non_time_columns
                    self.time_columns = []
                    # Drop time columns if not keeping them
                    df = df[non_time_columns]
            
            data_array = df.values
            total_rows = len(data_array)
            max_start = total_rows - window_length
            if max_start < 0:
                return

            self.all_full_sequences.append((data_array, action, file_path))
            # Compute stride using overlap (e.g., 0.5 => 50% overlap)
            stride = max(1, int(window_length * (1 - self.overlap)))
            for start_idx in range(0, max_start + 1, stride):
                window_data = data_array[start_idx : start_idx + window_length]
                self.samples.append((window_data, action))
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    def format_time_data(self, time_data):
        """
        Format time data for the Informer model which expects [month, day, weekday, hour, minute]
        Input: time_data is a numpy array of timestamps
        Output: formatted_time is a numpy array with shape [time_steps, 5]
        """
        # If time_data is None or empty, create dummy time data
        if time_data is None or len(time_data) == 0:
            # Create a sequence from 0 to time_steps
            time_steps = self.lag if time_data is None else len(time_data)
            # Create a synthetic time series with increasing values
            formatted_time = np.zeros((time_steps, 5), dtype=np.int64)
            
            for i in range(time_steps):
                # Format as [month, day, weekday, hour, minute]
                # Use simple patterns that vary across the time dimension
                formatted_time[i, 0] = i % 12  # Month (0-11)
                formatted_time[i, 1] = i % 28  # Day (0-27)
                formatted_time[i, 2] = i % 7   # Weekday (0-6)
                formatted_time[i, 3] = i % 24  # Hour (0-23)
                formatted_time[i, 4] = i % 60  # Minute (0-59)
        else:
            # Use the actual time data to create values
            time_steps = len(time_data)
            formatted_time = np.zeros((time_steps, 5), dtype=np.int64)
            
            # Create a mapping from the time values to Informer's expected format
            # Create an increasing sequence that maps well to the Informer model's expectations
            min_time = np.min(time_data)
            max_time = np.max(time_data)
            time_range = max(0.0001, max_time - min_time)  # Avoid division by zero
            
            for i in range(time_steps):
                # Normalize the time to [0, 1] range
                normalized_time = (time_data[i] - min_time) / time_range
                
                # Use the normalized time to create time features
                # These are all integer features in specific ranges
                formatted_time[i, 0] = int(normalized_time * 12) % 12    # Month (0-11)
                formatted_time[i, 1] = int(normalized_time * 28) % 28    # Day (0-27)
                formatted_time[i, 2] = int(normalized_time * 7) % 7      # Weekday (0-6)
                formatted_time[i, 3] = int(normalized_time * 24) % 24    # Hour (0-23)
                formatted_time[i, 4] = int(normalized_time * 60) % 60    # Minute (0-59)
                
        return formatted_time

    def _split_window(self, window_data):
        """
        Splits window_data into input (first 'lag' rows) and target (next 'n_ahead' rows),
        using a helper to filter sensor columns.

        The left half (columns 0-20) is used for input and the right half (columns 21-41) for target.
        If keep_time is True, time columns are preserved separately.
        """
        total_cols = len(self.processed_columns)
        if not self.keep_time:
            half = total_cols // 2  # Expecting first half and second half split

            def filter_columns(lower, upper, sensor):
                """Return indices between lower and upper that contain the sensor substring or all if 'all'."""
                if sensor == "all":
                    return list(range(lower, upper))
                # Filter based on if the column name contains the sensor substring.
                return [i for i in range(lower, upper) if sensor in self.processed_columns[i].lower()]

            input_cols = filter_columns(0, half, self.input_sensor)
            target_cols = filter_columns(half, total_cols, self.target_sensor)

            # Extract input arrays - shape [lag, num_input_channels]
            X = window_data[:self.lag, input_cols]
            # Extract target arrays - shape [n_ahead, num_target_channels]
            Y = window_data[self.lag : self.lag + self.n_ahead, target_cols]
            
            return X, Y
        else:
            # Handle case where time columns are kept
            # First identify non-time column indices
            non_time_cols = [i for i, col in enumerate(self.processed_columns) if col not in self.time_columns]
            time_cols = [i for i, col in enumerate(self.processed_columns) if col in self.time_columns]
            
            half = len(non_time_cols) // 2
            
            # Map the indices for first half (input) and second half (target) of non-time columns
            input_indices = non_time_cols[:half]
            target_indices = non_time_cols[half:]
            
            # Filter based on sensor type if needed
            if self.input_sensor != "all":
                input_indices = [i for i in input_indices if self.input_sensor in self.processed_columns[i].lower()]
            if self.target_sensor != "all":
                target_indices = [i for i in target_indices if self.target_sensor in self.processed_columns[i].lower()]
            
            # Extract input and target arrays
            X = window_data[:self.lag, input_indices]
            Y = window_data[self.lag:self.lag + self.n_ahead, target_indices]
            
            # Extract time data if available
            X_time_raw = window_data[:self.lag, time_cols] if time_cols else None
            Y_time_raw = window_data[self.lag:self.lag + self.n_ahead, time_cols] if time_cols else None
            
            # Format time data for Informer model
            X_time = self.format_time_data(X_time_raw) if X_time_raw is not None else None
            Y_time = self.format_time_data(Y_time_raw) if Y_time_raw is not None else None
            
            return X, Y, X_time, Y_time

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window_data, action = self.samples[idx]
        
        if not self.keep_time:
            X_array, Y_array = self._split_window(window_data)
            
            # Compute a weight based on the mean of the values.
            gt_avg = np.mean(Y_array)
            bin_value = self.value_to_bin_index(gt_avg, step=0.01, min_val=0, max_val=1)
            weight = self.weights[bin_value]
            
            return (
                torch.tensor(X_array, dtype=torch.float32),  # shape: [lag, in_channels]
                torch.tensor(Y_array, dtype=torch.float32),  # shape: [n_ahead, out_channels]
                action,
                weight,
            )
        else:
            X_array, Y_array, X_time, Y_time = self._split_window(window_data)
            
            # Compute a weight based on the mean of the values.
            gt_avg = np.mean(Y_array)
            bin_value = self.value_to_bin_index(gt_avg, step=0.01, min_val=0, max_val=1)
            weight = self.weights[bin_value]
            
            # Create proper time tensors - ensure they're the right shape
            batch_size = 1  # Single sample
            seq_len_x = self.lag
            seq_len_y = self.n_ahead
            time_features = 5  # [month, day, weekday, hour, minute]
            
            # Create empty time tensors with zeros if time data is None
            if X_time is None:
                X_time_tensor = torch.zeros((seq_len_x, time_features), dtype=torch.long)
            else:
                X_time_tensor = torch.tensor(X_time, dtype=torch.long)
                
            if Y_time is None:
                Y_time_tensor = torch.zeros((seq_len_y, time_features), dtype=torch.long)
            else:
                Y_time_tensor = torch.tensor(Y_time, dtype=torch.long)
            
            return (
                torch.tensor(X_array, dtype=torch.float32),  # shape: [lag, in_channels]
                torch.tensor(Y_array, dtype=torch.float32),  # shape: [n_ahead, out_channels]
                X_time_tensor,  # Now always a tensor with shape [lag, time_features]
                Y_time_tensor,  # Now always a tensor with shape [n_ahead, time_features]
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
        
        # Calculate statistics for better normalization
        X_mean = np.mean(X_array, axis=0, keepdims=True)
        X_std = np.std(X_array, axis=0, keepdims=True) + 1e-6  # Prevent division by zero
        
        # Z-score normalization instead of min-max for input
        X_array_norm = (X_array - X_mean) / X_std
        
        # For target, keep min-max but with a wider range to prevent clipping
        Y_min = np.min(Y_array, axis=0, keepdims=True)
        Y_max = np.max(Y_array, axis=0, keepdims=True)
        Y_range = Y_max - Y_min + 1e-8
        
        # Only normalize if there's actually a range to normalize
        if np.any(Y_range > 1e-6):
            Y_array_norm = 2 * ((Y_array - Y_min) / Y_range) - 1
        else:
            # If the range is too small, just center the data
            Y_array_norm = Y_array - Y_mean
        
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