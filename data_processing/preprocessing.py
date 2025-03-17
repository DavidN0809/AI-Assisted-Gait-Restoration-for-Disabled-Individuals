#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# -------------------------------
# Configuration for dataset paths and target frequency
# -------------------------------
class Configuration:
    def __init__(self, root, scaling, dataRange, target_rate):
        self.root = root
        self.raw = os.path.join(root)       # raw CSV files
        self.processed = os.path.join(root, "processed")
        self.scaling = scaling
        self.dr = dataRange
        self.target_rate = target_rate  # target resampling frequency in Hz

#base_dir=r"/data1/dnicho26/EMG_DATASET/data"
base_dir=r"D:\UNC Charlotte Dropbox\orgs-ecgr-QuantitativeImagingandAILaboratory"
config = Configuration(root=base_dir, scaling=10000, dataRange=slice(0, -1), target_rate=148.148)

# Set this flag to True to run in index-only mode
index_only = False

# -------------------------------
# Helper: Find CSV files in the directory structure
# -------------------------------
def find_csv_files(directory):
    csv_files = list(Path(directory).rglob("*.csv"))
    return [str(f) for f in csv_files]

# -------------------------------
# Helper: Rename sensor columns
# -------------------------------
def rename_sensor_columns(df):
    rename_map = {}
    for col in df.columns:
        # Example renaming: prepend 'EMG_sensor' for columns containing 'EMG'
        # and 'Accel_sensor' for those containing 'Acc'. Adjust as needed.
        if "EMG" in col:
            new_name = col.strip().replace("EMG", "EMG_sensor")
            rename_map[col] = new_name
        elif "Acc" in col:
            new_name = col.strip().replace("Acc", "Accel_sensor")
            rename_map[col] = new_name
    return df.rename(columns=rename_map)

# -------------------------------
# Update index CSV with processed file paths and number of frames
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
# Data Preprocessing Functions
# -------------------------------
def normalize(x):
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return np.zeros_like(x)
    return (x - mean) / std

def standardize(x):
    m = np.max(x)
    return x / m if m != 0 else x

def replace_outliers_zscore(x, threshold=3):
    std = np.std(x)
    if std == 0:
        return x.copy()
    z = np.abs((x - np.mean(x)) / std)
    x_filtered = x.copy()
    x_filtered[z >= threshold] = np.median(x)
    return x_filtered

def resample_signal(signal, orig_rate, target_rate):
    """
    Resamples a 1D signal from the original sampling rate to the target sampling rate
    using linear interpolation.
    """
    duration = len(signal) / orig_rate
    new_length = int(np.floor(duration * target_rate))
    original_time = np.linspace(0, duration, len(signal))
    target_time = np.linspace(0, duration, new_length)
    return np.interp(target_time, original_time, signal)

# -------------------------------
# Process a sensor series (vectorized) using resampling instead of a sliding window
# -------------------------------
# Regular expression to extract frequency from column header, e.g. "(1259.259 Hz)"
freq_pattern = re.compile(r'\(([\d.]+)\s*Hz\)')

def process_series(s, target_rate):
    arr = s.astype(float).values
    # Attempt to extract the original frequency from the column header.
    m = freq_pattern.search(s.name)
    if m:
        orig_rate = float(m.group(1))
    else:
        # If no frequency info is available, assume it is already at target_rate.
        orig_rate = target_rate

    # Step 1: Replace outliers using a z-score threshold.
    arr = replace_outliers_zscore(arr, threshold=3)
    # Step 2: Subtract the mean and take the absolute value.
    arr = np.abs(arr - np.mean(arr))
    # Step 3: Resample the signal to the target frequency.
    arr = resample_signal(arr, orig_rate, target_rate)
    # Step 4: Standardize by dividing by the maximum value.
    arr = standardize(arr)
    # Create a new Series with a new integer index matching the resampled length.
    return pd.Series(arr, index=range(len(arr)), dtype=float)

# -------------------------------
# Function to process a single CSV file
# -------------------------------
def process_file(file_path, raw_dir, processed_dir):
    # Read and clean the CSV file
    df = pd.read_csv(file_path, header=0)
    df.columns = df.columns.astype(str).str.strip()
    
    # Remove extra time columns: keep only "time 0" then rename it to "time"
    time_cols = [col for col in df.columns if col.lower().startswith("time") and col.lower() != "time 0"]
    if time_cols:
        df.drop(columns=time_cols, inplace=True)
    if "time 0" in df.columns:
        df.rename(columns={"time 0": "time"}, inplace=True)
    
    # Ensure time column is numeric
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df.fillna(0, inplace=True)

    # Process sensor columns (all except 'time') – also apply scaling if needed
    sensor_cols = df.columns.difference(['time'])
    df[sensor_cols] = df[sensor_cols].astype(float) * config.scaling

    # Process each sensor series using the resampling method
    df_processed = df.copy()
    # Instead of applying a sliding window average, we resample each series
    processed_sensor = {}
    for col in sensor_cols:
        processed_sensor[col] = process_series(df[col], config.target_rate)
    # When combining the processed columns, align by the resampled index (they may differ in length).
    # For simplicity, we take the minimum length across all processed sensor columns.
    min_length = min(len(s) for s in processed_sensor.values())
    for col in processed_sensor:
        processed_sensor[col] = processed_sensor[col].iloc[:min_length]
    # Reconstruct the processed dataframe.
    df_processed = pd.DataFrame(processed_sensor)
    
    # Optionally: Rename sensor columns according to your old naming convention
    df_processed = rename_sensor_columns(df_processed)
    
    # Reconstruct relative directory structure so processed files mirror raw tree.
    rel_path = os.path.relpath(file_path, raw_dir)
    rel_dir = os.path.dirname(rel_path)
    output_dir_full = os.path.join(processed_dir, rel_dir)
    os.makedirs(output_dir_full, exist_ok=True)

    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    output_csv = os.path.join(output_dir_full, f"{base_filename}.csv")
    df_processed.to_csv(output_csv, index=False, float_format="%.5f")
    return output_csv

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
            futures = {
                executor.submit(process_file, file, raw_dir, processed_dir): file
                for file in csv_files
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing CSV files", unit="file"):
                try:
                    result = future.result()
                    if result is not None:
                        processed_files.append(result)
                except Exception as exc:
                    print(f"Error processing file: {exc}")

        print(f"Successfully processed {len(processed_files)} files.")
        update_index(processed_files, processed_dir)