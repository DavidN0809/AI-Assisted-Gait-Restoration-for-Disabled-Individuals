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


def manual_min_max_scale(vals, feature_range=(-1, 1)):
    a, b = feature_range
    mn = np.min(vals)
    mx = np.max(vals)
    if mx == mn:
        return np.zeros_like(vals)
    norm = (vals - mn) / (mx - mn)
    return norm * (b - a) + a

class EMG_dataset(torch.utils.data.Dataset):
    """
    Sliding-window dataset for EMG/ACC forecasting.
    If keep_time=True, returns real time-mark tensors alongside inputs/targets.
    Non-overlapping windows: hop = lag + n_ahead.
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
        self.time_columns = []

        # Load and process files
        if self.single_file_mode:
            self._process_file(processed_index_csv, "unknown")
        else:
            idx_df = pd.read_csv(processed_index_csv)
            for _, row in idx_df.iterrows():
                path = row["file_path"]
                if path.startswith("./"):
                    path = path[2:]
                full = os.path.join(self.base_dir, path)
                self._process_file(full, row.get("action", "unknown"))

        if self.processed_columns is None:
            raise RuntimeError("No files processed; processed_columns is None.")
        # Index of numeric (non-time) columns for distribution
        self.non_time_idx = [i for i, col in enumerate(self.processed_columns)
                             if col not in self.time_columns]

        print(f"Total windows: {len(self.samples)}")
        self.get_distribution()

    def _process_file(self, file_path, action):
        window_len = self.lag + self.n_ahead
        if not os.path.isfile(file_path):
            return
        df = pd.read_csv(file_path).dropna(axis=0)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        if df.empty:
            return

        # Detect time cols
        tcols = [c for c in df.columns if 'time' in c.lower()]
        if self.processed_columns is None:
            if self.keep_time and tcols:
                df[tcols] = df[tcols].apply(lambda col: pd.to_datetime(col.astype(float), unit='s', origin='unix'))
                self.time_columns = tcols
                self.processed_columns = df.columns.tolist()
            else:
                self.processed_columns = [c for c in df.columns if c not in tcols]
                df = df[self.processed_columns]

        data = df.values
        total = len(data)
        # Non-overlapping hop
        stride = window_len if self.overlap == 0 else max(1, int(window_len * (1 - self.overlap)))
        for start in range(0, total - window_len + 1, stride):
            wnd = data[start:start + window_len]
            self.samples.append((wnd, action))

    def format_time_data(self, time_slice):
        arr = time_slice.flatten().astype('datetime64[ns]')
        dt = pd.DatetimeIndex(arr)
        mat = np.stack([dt.month, dt.day, dt.weekday, dt.hour, dt.minute], axis=1)
        return mat.reshape(time_slice.shape[0], 5)


    def _split_window(self, wnd):
        """
        Split a single window into inputs and targets, optionally with time.
        """
        cols = self.processed_columns
        # Get time column indices
        t_idx = [cols.index(c) for c in self.time_columns]
        # Non-time numeric columns
        non_t = [i for i in range(len(cols)) if i not in t_idx]
        total_nt = len(non_t)
        half = total_nt // 2
        # Input and target numeric halves
        input_nt = non_t[:half]
        target_nt = non_t[half:]
        # Filter by sensor substring
        inp_idx = [i for i in input_nt if self.input_sensor=='all' or self.input_sensor in cols[i].lower()]
        tgt_idx = [i for i in target_nt if self.target_sensor=='all' or self.target_sensor in cols[i].lower()]
        # Slice numeric sequences
        X = wnd[:self.lag, inp_idx]
        Y = wnd[self.lag:self.lag + self.n_ahead, tgt_idx]
        if not self.keep_time:
            return X, Y
        # Otherwise build time marks
        Xt = wnd[:self.lag, t_idx] if t_idx else None
        Yt = wnd[self.lag:self.lag + self.n_ahead, t_idx] if t_idx else None
        X_time = self.format_time_data(Xt) if Xt is not None else None
        Y_time = self.format_time_data(Yt) if Yt is not None else None
        return X, Y, X_time, Y_time
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wnd, action = self.samples[idx]
        if not self.keep_time:
            X, Y = self._split_window(wnd)
            X = np.asarray(X, dtype=np.float32)
            Y = np.asarray(Y, dtype=np.float32)
            w = self._compute_weight(Y)
            return (
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(Y, dtype=torch.float32),
                action,
                torch.tensor(w, dtype=torch.float32)
            )
        else:
            X, Y, Xt, Yt = self._split_window(wnd)
            X = np.asarray(X, dtype=np.float32)
            Y = np.asarray(Y, dtype=np.float32)
            w = self._compute_weight(Y)
            X_time_arr = Xt
            Y_time_arr = Yt
            return (
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(Y, dtype=torch.float32),
                torch.tensor(X_time_arr, dtype=torch.long),
                torch.tensor(Y_time_arr, dtype=torch.long),
                action,
                torch.tensor(w, dtype=torch.float32)
            )

    def _compute_weight(self, Y):
        avg = np.mean(Y)
        idx = self.value_to_bin_index(avg, step=0.01, min_val=0, max_val=1)
        return float(self.weights[idx])

    def get_distribution(self):
        bins = np.arange(0, 1.01, 0.01)
        agg = np.zeros(len(bins) - 1, dtype=int)
        for wnd, _ in tqdm(self.samples, desc="Computing Distribution"):
            vals = wnd[:, self.non_time_idx].astype(float).ravel()
            counts, _ = np.histogram(vals, bins=bins)
            agg += counts
        self.bin_centers = (bins[:-1] + bins[1:]) / 2
        inv = 1 / (agg + 1e-6)
        self.weights = manual_min_max_scale(inv, feature_range=(0, 1))
        self.distribution = agg

    def value_to_bin_index(self, x, step=0.01, min_val=0, max_val=1):
        x = max(min_val, min(x, max_val))
        num = int((max_val - min_val) / step)
        idx = int((x - min_val) // step)
        return min(idx, num - 1)

    def plot_distribution(self, fig_path):
        plt.figure(figsize=(10, 6))
        plt.bar(self.bin_centers, self.distribution, width=0.005, align='center')
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
    Each window is normalized by channel (per column) for both input and target data.
    You can optionally override the base_dir by passing trial_dir.
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
        trial_dir: str = None
    ):
        # if user passes a specific trial_dir, use that instead of base_dir
        if trial_dir is not None:
            base_dir = trial_dir
        # forward all args exactly to EMG_dataset
        super().__init__(
            processed_index_csv,
            lag,
            n_ahead,
            overlap=overlap,
            input_sensor=input_sensor,
            target_sensor=target_sensor,
            base_dir=base_dir,
            keep_time=keep_time,
            single_file_mode=single_file_mode
        )

    def __getitem__(self, idx):
        # grab the raw window
        window_data, action = self.samples[idx]
        X_array, Y_array = self._split_window(window_data)

        # per-window min–max scaling to [0,1]
        # input
        X_min = X_array.min(axis=0, keepdims=True)
        X_max = X_array.max(axis=0, keepdims=True)
        X_range = X_max - X_min + 1e-8
        X_norm = (X_array - X_min) / X_range

        # target
        Y_min = Y_array.min(axis=0, keepdims=True)
        Y_max = Y_array.max(axis=0, keepdims=True)
        Y_range = Y_max - Y_min + 1e-8
        Y_norm = (Y_array - Y_min) / Y_range

        # compute weight from original (unnormalized) Y
        gt_avg = Y_array.mean()
        bin_value = self.value_to_bin_index(gt_avg, step=0.01, min_val=0, max_val=1)
        weight = float(self.weights[bin_value])

        return (
            torch.tensor(X_norm, dtype=torch.float32),
            torch.tensor(Y_norm, dtype=torch.float32),
            action,
            torch.tensor(weight, dtype=torch.float32),
        )
