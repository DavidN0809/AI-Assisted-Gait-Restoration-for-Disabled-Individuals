import os
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from fractions import Fraction
from scipy.signal import butter, filtfilt, iirnotch, resample_poly
from tqdm import tqdm
import concurrent.futures  # New import for threading

# --------------------------
# 1. CSV Processing Functions
# --------------------------

def process_csv_file(input_csv, output_csv):
    """
    Reads a CSV file, drops any columns that do not contain 'emg' (case-insensitive)
    in the column name, and applies EMG preprocessing to the remaining columns.
    Saves the resulting DataFrame to output_csv.
    """
    df = pd.read_csv(input_csv)
    df = df.dropna()
    # Drop first column if its header is empty
    if df.columns[0].strip() == "":
        df.drop(columns=df.columns[0], inplace=True)
    
    # Drop all columns that do NOT contain "emg" (case-insensitive)
    df = df.loc[:, df.columns.str.contains("emg", case=False)]
    
    # Preprocess each remaining EMG column
    for col in df.columns:
        df[col] = preprocess_emg_no_resample(df[col])
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Processed CSV saved to: {output_csv}")
    return df

def process_all_csvs(input_base_dir, output_base_dir):
    """
    Recursively traverse input_base_dir for CSV files (ignoring directories with 'camera_')
    and process them sequentially with process_csv_file.
    Processed CSVs are saved to output_base_dir preserving folder structure.
    """
    for root, dirs, files in os.walk(input_base_dir):
        # Skip directories if any part of the relative path starts with 'camera_'
        rel_parts = os.path.relpath(root, input_base_dir).split(os.sep)
        if any(part.lower().startswith("camera_") for part in rel_parts):
            continue
        for file in files:
            if file.lower().endswith('.csv'):
                input_csv = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_base_dir)
                output_dir = os.path.join(output_base_dir, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                output_csv = os.path.join(output_dir, "processed_" + file)
                process_csv_file(input_csv, output_csv)

def process_all_csvs_threaded(input_base_dir, output_base_dir):
    """
    Recursively traverse input_base_dir for CSV files (ignoring directories with 'camera_')
    and process them in parallel using os.cpu_count() workers.
    Processed CSVs are saved to output_base_dir preserving folder structure.
    """
    tasks = []
    for root, dirs, files in os.walk(input_base_dir):
        rel_parts = os.path.relpath(root, input_base_dir).split(os.sep)
        if any(part.lower().startswith("camera_") for part in rel_parts):
            continue
        for file in files:
            if file.lower().endswith('.csv'):
                input_csv = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_base_dir)
                output_dir = os.path.join(output_base_dir, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                output_csv = os.path.join(output_dir, "processed_" + file)
                tasks.append((input_csv, output_csv))
                
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_csv_file, task[0], task[1]) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing file: {e}")

# --------------------------
# 2. EMG Preprocessing Functions (No Resampling)
# --------------------------

def replace_outliers_zscore(x, threshold=3):
    """
    Replace values with absolute z-score exceeding threshold by the median.
    """
    mean_val = np.nanmean(x)
    std_val = np.nanstd(x)
    if std_val == 0:
        return x
    z = np.abs((x - mean_val) / std_val)
    median_val = np.nanmedian(x)
    x[z >= threshold] = median_val
    return x

def bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def notch_filter(data, freq, fs, Q=30):
    w0 = freq / (0.5 * fs)
    b, a = iirnotch(w0, Q)
    return filtfilt(b, a, data)

def causal_moving_average(signal, window=3):
    out = np.zeros_like(signal)
    cum_sum = 0
    for i in range(len(signal)):
        cum_sum += signal[i]
        if i >= window:
            cum_sum -= signal[i - window]
            out[i] = cum_sum / window
        else:
            out[i] = cum_sum / (i + 1)
    return out

def manual_min_max_scale(vals, feature_range=(-1, 1)):
    min_val = np.min(vals)
    max_val = np.max(vals)
    if max_val == min_val:
        return np.zeros_like(vals)
    a, b = feature_range
    normalized = (vals - min_val) / (max_val - min_val) * (b - a) + a
    return normalized

def preprocess_emg_no_resample(emg_signal, fs=1259.259, 
                               lowcut=10, highcut=500, notch_freq=50, window=3):
    """
    Preprocess raw EMG data (sampled at 1259.259 Hz) without resampling.
    Steps:
      - Remove outliers using Z-score method.
      - (Optional) Apply a bandpass filter (10–500 Hz) and a notch filter (50 Hz).
      - Rectify, smooth (causal moving average), and Z-score normalize.
      - Scale normalized data to the range [-1, 1].
    """
    emg_clean = replace_outliers_zscore(emg_signal, threshold=3)
    # Note: filtering steps are commented out. Uncomment if needed.
    # filtered = bandpass_filter(emg_clean, lowcut, highcut, fs, order=2)
    # filtered = notch_filter(filtered, freq=notch_freq, fs=fs, Q=30)
    # rectified = np.abs(filtered)
    # envelope = causal_moving_average(rectified, window=window)

    # Scale normalized data to the range [-1, 1]
    scaled_emg = manual_min_max_scale(emg_clean, feature_range=(-1, 1))
    return scaled_emg

# --------------------------
# 3. Sliding Window Segmentation
# --------------------------

def segment_signal(signal, window_size, step_size):
    """
    Segments a 1D signal into overlapping windows.
    Returns an array of shape (num_windows, window_size).
    """
    num_samples = signal.shape[0]
    segments = []
    for start in range(0, num_samples - window_size + 1, step_size):
        segments.append(signal[start:start+window_size])
    return np.array(segments)

# --------------------------
# 4. Index Building for Train/Val/Test Splits
# --------------------------

def build_index(processed_csv_files, seed=42, train_ratio=0.7, val_ratio=0.15):
    """
    Given a list of processed CSV file paths, extract metadata (UUID and action from path)
    and randomly assign each file to train, validation, or test splits.
    Returns three DataFrames: train_idx, val_idx, test_idx.
    """
    random.seed(seed)
    index_list = []
    for file in processed_csv_files:
        # Assuming file path: .../<uuid>/<action>/processed_<file>.csv
        parts = os.path.normpath(file).split(os.sep)
        try:
            uuid = parts[-3]
            action = parts[-2]
        except IndexError:
            uuid = "unknown"
            action = "unknown"
        index_list.append({
            'file_path': file,
            'uuid': uuid,
            'action': action
        })
    df_index = pd.DataFrame(index_list)
    files = df_index['file_path'].tolist()
    random.shuffle(files)
    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_files = files[:n_train]
    val_files = files[n_train:n_train+n_val]
    test_files = files[n_train+n_val:]
    
    train_idx = df_index[df_index['file_path'].isin(train_files)]
    val_idx = df_index[df_index['file_path'].isin(val_files)]
    test_idx = df_index[df_index['file_path'].isin(test_files)]
    
    return train_idx, val_idx, test_idx

def save_index(index_df, output_csv):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    index_df.to_csv(output_csv, index=False)
    print(f"Index saved to: {output_csv}")

# --------------------------
# 6. Main Integration
# --------------------------

def main():
    input_base_dir = "/data1/dnicho26/EMG_DATASET/data/processed-server"
    output_base_dir = "/data1/dnicho26/EMG_DATASET/data/final-data"
    figures_dir = "/data1/dnicho26/Thesis/AI-Assisted-Gait-Restoration-for-Disabled-Individuals/figures/processing"
    index_output_dir = os.path.join(output_base_dir, "index_files")
    
    # Process all CSV files in parallel using threading
    print("Processing all CSV files in parallel...")
    process_all_csvs_threaded(input_base_dir, output_base_dir)
    
    # Build index from processed CSV files
    processed_csvs = glob(os.path.join(output_base_dir, "**", "*.csv"), recursive=True)
    print(f"\nTotal processed CSV files found: {len(processed_csvs)}")
    train_idx, val_idx, test_idx = build_index(processed_csvs, seed=42)
    os.makedirs(index_output_dir, exist_ok=True)
    save_index(train_idx, os.path.join(index_output_dir, "train_index.csv"))
    save_index(val_idx, os.path.join(index_output_dir, "val_index.csv"))
    save_index(test_idx, os.path.join(index_output_dir, "test_index.csv"))
    
if __name__ == "__main__":
    main()
