#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# -------------------------------
# Configuration for dataset paths
# -------------------------------
class Configuration:
    def __init__(self, root, scaling, dataRange):
        self.root = root
        self.raw = os.path.join(root, "data")
        self.processed = os.path.join(root, "processed")
        self.scaling = scaling
        self.dr = dataRange

config = Configuration(root=r"/data1/dnicho26/EMG_DATASET", scaling=10000, dataRange=slice(0, -1))

# Set this flag to True to run in index-only mode
index_only = True

# -------------------------------
# Helper: Find CSV files in the directory structure
# -------------------------------
def find_csv_files(directory):
    csv_files = list(Path(directory).rglob("*.csv"))
    return [str(f) for f in csv_files]

# -------------------------------
# Update index CSV with processed file paths
# -------------------------------
def update_index(processed_files, processed_dir):
    index_file = os.path.join(processed_dir, "index.csv")
    records = []
    
    for f in tqdm(processed_files, desc="Indexing processed files", unit="file"):
        try:
            df = pd.read_csv(f)
            frames = df.shape[0]  # Number of rows (frames)
            records.append({"emg_file": f, "frames": frames})
        except Exception as e:
            print(f"Error reading {f}: {e}")

    df_index = pd.DataFrame(records)
    df_index.to_csv(index_file, index=False)
    print(f"Index updated: {index_file}")

# -------------------------------
# Function to process a single CSV file
# -------------------------------
def process_file(file_path, raw_dir, processed_dir, window_size=None):
    df = pd.read_csv(file_path, header=0)
    df.columns = df.columns.astype(str).str.strip()
    
    # Drop all time columns except "time 0" and rename "time 0" to "time"
    time_cols = [col for col in df.columns if col.lower().startswith("time") and col.lower() != "time 0"]
    if time_cols:
        df.drop(columns=time_cols, inplace=True)
    if "time 0" in df.columns:
        df.rename(columns={"time 0": "time"}, inplace=True)
    
    # Convert time column to numeric and fill NaNs.
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df.fillna(0, inplace=True)

    # Process sensor columns (all columns except 'time')
    sensor_cols = df.columns.difference(['time'])
    df[sensor_cols] = df[sensor_cols].astype(float)

    # Copy dataframe for processing
    df_processed = df.copy()
    df_processed.loc[:, sensor_cols] = df.loc[:, sensor_cols].apply(lambda s: process_series(s, window_size), axis=0)

    # Reconstruct relative directory structure
    rel_path = os.path.relpath(file_path, raw_dir)
    rel_dir = os.path.dirname(rel_path)
    output_dir_full = os.path.join(processed_dir, rel_dir)
    os.makedirs(output_dir_full, exist_ok=True)

    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    output_csv = os.path.join(output_dir_full, f"{base_filename}.csv")
    df_processed.to_csv(output_csv, index=False, float_format="%.5f")
    return output_csv

# -------------------------------
# Process a sensor series (vectorized)
# -------------------------------
def process_series(s, window_size=None):
    arr = s.astype(float).values
    arr = replace_outliers_zscore(arr)
    arr = normalize(arr)
    arr = standardize(arr)
    if window_size is not None and window_size > 1:
        arr = average_window(arr, window_size)
    return pd.Series(arr, index=s.index, dtype=float)

# -------------------------------
# Data Preprocessing Functions
# -------------------------------
def normalize(x):
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return np.zeros_like(x)
    return (x - mean) / std

def standardize(x):
    m = max(x)
    return x / m if m != 0 else x

def replace_outliers_zscore(x, threshold=3):
    std = np.std(x)
    if std == 0:
        return x.copy()
    z = np.abs((x - np.mean(x)) / std)
    x_filtered = x.copy()
    x_filtered[z >= threshold] = np.median(x)
    return x_filtered

def average_window(signal_array, window_size, averaging_method=np.average):
    moving_averages = []
    for i in range(window_size, len(signal_array) + 1, window_size):
        window = signal_array[i-window_size:i]
        avg = averaging_method(window)
        moving_averages.append(avg)
    return np.array(moving_averages)

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    processed_dir = config.processed
    os.makedirs(processed_dir, exist_ok=True)

    if index_only:
        print("Running in index-only mode. Scanning processed directory...")
        processed_files = find_csv_files(processed_dir)
        update_index(processed_files, processed_dir)
    else:
        print("Processing raw data and updating index...")
        raw_dir = config.raw
        csv_files = find_csv_files(raw_dir)
        print(f"Found {len(csv_files)} CSV files in {raw_dir}")

        processed_files = []
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {executor.submit(process_file, file, raw_dir, processed_dir): file for file in csv_files}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing CSV files", unit="file"):
                result = future.result()
                if result is not None:
                    processed_files.append(result)

        print(f"Successfully processed {len(processed_files)} files.")
        update_index(processed_files, processed_dir)
