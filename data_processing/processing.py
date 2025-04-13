#!/usr/bin/env python3

import os
import glob
import pandas as pd
import numpy as np
import random
import re
import scipy.signal
import concurrent.futures
from tqdm import tqdm

def replace_outliers(series, threshold=3):
    """Replace values that deviate more than threshold*std from the mean with the median."""
    values = series.values.astype(np.float64)
    mean = np.mean(values)
    std = np.std(values)
    median = np.median(values)
    mask = np.abs(values - mean) > threshold * std
    values[mask] = median
    return pd.Series(values, index=series.index)


def parse_frequency_from_column(col_name):
    """Parse frequency from a column name in the format '(1000 Hz)'."""
    pattern = re.compile(r'\(([\d\.]+)\s*Hz\)', re.IGNORECASE)
    match = pattern.search(col_name)
    if match:
        try:
            return float(match.group(1))
        except:
            return None
    return None


def normalize_action_name(action_name):
    """
    Normalize action names by removing underscores, standardizing spaces, 
    and converting to lowercase to ensure consistent action identification.
    """
    # Convert to lowercase
    action = action_name.lower()
    
    # Replace underscores and multiple spaces with a single space
    action = action.replace('_', ' ')
    action = ' '.join(action.split())
    
    # Remove any trailing or leading spaces
    action = action.strip()
    
    return action


def preprocess_csv_file(input_csv, input_base_dir, output_base_dir, target_hz=10, outlier_threshold=3):
    """
    Reads a CSV of rectified EMG signals, replaces outliers, normalizes each column between 0 and 1,
    parses the source frequency from each column name (default 1000 Hz if not found), keeps time columns,
    and resamples each column individually to a common length (minimum length computed from each column's
    new length = int(len(column) * (target_hz / source_hz))).
    Saves the processed file preserving the relative path.
    """

    data = pd.read_csv(input_csv)


    # Identify time columns but keep them
    time_columns = [col for col in data.columns if 'time' in col.lower()]
    non_time_columns = [col for col in data.columns if 'time' not in col.lower()]
    
    processed_cols = {}  
    new_lengths = []
    
    # Process non-time columns
    for col in non_time_columns:
        # Process each column separately
        series = data[col].dropna().astype(np.float64)
        
        # 1. Replace outliers
        series = replace_outliers(series, threshold=outlier_threshold)
        
        # 2. Calculate resampling parameters
        f_source = parse_frequency_from_column(col)
        if f_source is None:
            f_source = 1000  
        n_rows = len(series)
        new_length = int(n_rows * (target_hz / f_source))
        if new_length < 1:
            new_length = 1
        new_lengths.append(new_length)
        processed_cols[col] = series

    if not new_lengths:
        print(f"No columns to process in {input_csv} after identifying time columns.")
        return
    
    # Find common length for resampling
    common_length = min(new_lengths)
    
    # 2. Resample all non-time columns to the common length
    resampled_data = {}
    for col, series in processed_cols.items():
        resampled_array = scipy.signal.resample(series.to_numpy(), common_length)
        resampled_data[col] = resampled_array
    
    # Also resample time columns if they exist
    for col in time_columns:
        series = data[col].dropna().astype(np.float64)
        # Resample time column to the same length as other columns
        resampled_array = scipy.signal.resample(series.to_numpy(), common_length)
        resampled_data[col] = resampled_array

    # Create DataFrame from resampled data
    df_resampled = pd.DataFrame(resampled_data)

    # Drop the first 200 samples if there are enough samples
    if len(df_resampled) > 200:
        df_resampled = df_resampled.iloc[200:]
    else:
        print(f"File too short to drop samples: {len(df_resampled)} samples")

    # # # 3. Normalize each column after resampling
    # normalized_data = {}
    # for col in df_resampled.columns:
    #     series = df_resampled[col]
    #     col_min = series.min()
    #     col_max = series.max()
    #     if col_max - col_min != 0:
    #         normalized_data[col] = (series - col_min) / (col_max - col_min)
    #     else:
    #         normalized_data[col] = pd.Series(np.zeros_like(series), index=series.index)
    
    # Create final DataFrame with normalized data
    df_normalized = pd.DataFrame(normalized_data)
    # Save the processed file
    rel_path = os.path.relpath(input_csv, start=input_base_dir)
    output_csv = os.path.join(output_base_dir, rel_path)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_normalized.to_csv(output_csv, index=False)
    print(f"Saved processed file to {output_csv}. Common length: {common_length}.")


def process_all_csvs(input_base_dir, output_base_dir, target_hz=10, outlier_threshold=3, num_workers=4):
    """
    Finds all CSV files under input_base_dir, preprocesses them, and saves to output_base_dir.
    Uses parallel processing with specified number of workers.
    """
    csv_files = glob.glob(os.path.join(input_base_dir, "**", "*.csv"), recursive=True)
    print(f"Found {len(csv_files)} CSV files in {input_base_dir}")
    
    # Create output directories if they don't exist
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Create a list of future tasks
        futures = [
            executor.submit(
                preprocess_csv_file, 
                csv_file, 
                input_base_dir, 
                output_base_dir, 
                target_hz, 
                outlier_threshold
            ) 
            for csv_file in csv_files
        ]
        
        # Process futures as they complete with a progress bar
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing CSV files"):
            pass
            
    print("Preprocessing complete.")


def build_train_test_val_index(output_base_dir, train_ratio=0.6, val_ratio=0.2, seed=42):
    """
    Creates train/test/validation indexes for all processed CSV files in output_base_dir
    and saves them as separate CSV files.
    
    Args:
        output_base_dir (str): Base directory containing processed CSV files
        train_ratio (float): Ratio of files for training set (default 0.6)
        val_ratio (float): Ratio of files for validation set (default 0.2)
        seed (int): Random seed (not used when UUIDs are sorted)
    """
    all_files = glob.glob(os.path.join(output_base_dir, "**", "*.csv"), recursive=True)
    all_files = [os.path.relpath(f, start=output_base_dir) for f in all_files]
    
    if not all_files:
        print(f"No CSV files found in {output_base_dir}")
        return
    
    # Group files by UUID - extract UUID from the file path
    uuid_files = {}
    
    for file_path in all_files:
        # Split path into components
        path_parts = file_path.split(os.sep)
        
        # The UUID should be the first part of the path (e.g., '2' in '2/treadmill/...')
        uuid = path_parts[0]
        
        # The action is the second part of the path (e.g., 'treadmill' in '2/treadmill/...')
        action = path_parts[1] if len(path_parts) > 1 else ""
        
        # Normalize the action name
        action = normalize_action_name(action)
        
        # Add the action as additional data to the file info
        file_info = {"file_path": file_path, "action": action}
        
        if uuid not in uuid_files:
            uuid_files[uuid] = []
        uuid_files[uuid].append(file_info)
    
    # Print sample UUIDs for verification
    print(f"Sample UUIDs and actions: {list(uuid_files.items())[:3]}")
    
    # Get unique UUIDs and sort them (no shuffling)
    uuids = sorted(uuid_files.keys())
    
    # Calculate split sizes
    n_total = len(uuids)
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))
    
    # Ensure we don't exceed total with rounding
    if n_train + n_val > n_total:
        n_val = max(1, n_total - n_train - 1)  # Ensure at least 1 for test
    
    n_test = n_total - n_train - n_val
    
    # Sequential split: first n_train UUIDs go to train, next n_test to test, last n_val to validation
    train_uuids = uuids[:n_train]
    test_uuids = uuids[n_train:n_train + n_test]
    val_uuids = uuids[n_train + n_test:]
    
    # Print information about the splits
    print(f"Split {n_total} UUIDs: {n_train} train, {n_test} test, {n_val} validation")
    
    # Create dataframes
    train_files_info = []
    val_files_info = []
    test_files_info = []
    
    # Assign files to splits based on UUID
    for uuid, files_info in uuid_files.items():
        if uuid in train_uuids:
            train_files_info.extend(files_info)
        elif uuid in test_uuids:
            test_files_info.extend(files_info)
        else:
            val_files_info.extend(files_info)
    
    # Create dataframes
    train_index = pd.DataFrame(train_files_info)
    val_index = pd.DataFrame(val_files_info)
    test_index = pd.DataFrame(test_files_info)
    
    # Ensure output directories exist
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Save indexes to the output directory itself, not its parent
    train_index.to_csv(os.path.join(output_base_dir, "train_index.csv"), index=False)
    val_index.to_csv(os.path.join(output_base_dir, "val_index.csv"), index=False)
    test_index.to_csv(os.path.join(output_base_dir, "test_index.csv"), index=False)
    
    print(f"Saved train index to {os.path.join(output_base_dir, 'train_index.csv')} ({len(train_files_info)} files)")
    print(f"Saved validation index to {os.path.join(output_base_dir, 'val_index.csv')} ({len(val_files_info)} files)")
    print(f"Saved test index to {os.path.join(output_base_dir, 'test_index.csv')} ({len(test_files_info)} files)")


def process_data(input_dir, output_dir, target_hz=10, outlier_threshold=3, train_ratio=0.6, val_ratio=0.2, seed=42, 
                index_only=False, num_workers=4):
    """
    Process rectified EMG CSV files using RMS resampling based on parsed frequency,
    with outlier replacement, and create train/test/validation splits based on UUID.
    
    Args:
        input_dir (str): Input directory containing raw CSV files
        output_dir (str): Output directory to save processed CSV files
        target_hz (float): Target sampling frequency after resampling (Hz)
        outlier_threshold (float): Threshold for outlier replacement (in standard deviations)
        train_ratio (float): Ratio of files for training set (default 0.6)
        val_ratio (float): Ratio of files for validation set (default 0.2)
        seed (int): Random seed for splitting
        index_only (bool): If True, only build the index files without processing CSVs
        num_workers (int): Number of worker threads for parallel processing
    """
    if not index_only:
        process_all_csvs(input_dir, output_dir, target_hz, outlier_threshold, num_workers)
    
    build_train_test_val_index(output_dir, train_ratio, val_ratio, seed)


if __name__ == "__main__":
    # Configuration variables
    input_dir = "/data1/dnicho26/EMG_DATASET/data/processed"
    output_dir = "/data1/dnicho26/EMG_DATASET/final-data-not-norm/"
    target_hz = 10
    outlier_threshold = 3
    train_ratio = 0.6  # Updated to 60%
    val_ratio = 0.2    # Updated to 20%
    seed = 42
    index_only = False
    num_workers = 4

    process_data(
        input_dir, 
        output_dir, 
        target_hz, 
        outlier_threshold, 
        train_ratio, 
        val_ratio, 
        seed,
        index_only,
        num_workers
    )