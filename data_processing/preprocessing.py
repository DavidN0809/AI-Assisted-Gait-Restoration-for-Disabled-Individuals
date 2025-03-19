#!/usr/bin/env python3
import os
import re
import sys
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import random

# -------------------------------
# Minimal Configuration for dataset paths and splits
# -------------------------------
class Configuration:
    def __init__(self, root, train_split=0.7, val_split=0.10, test_split=0.20):
        self.root = root
        self.raw = os.path.join(root, "data")        # raw CSV files
        self.processed = os.path.join(root, "processed-test")
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

# Update this to your base directory.
base_dir = r"D:\UNC Charlotte Dropbox\orgs-ecgr-QuantitativeImagingandAILaboratory"
config = Configuration(root=base_dir)
# -------------------------------
# Helper: Find CSV files in the directory structure
# -------------------------------
def find_csv_files(directory):
    return [str(f) for f in Path(directory).rglob("*.csv")]

# -------------------------------
# Sensor Name Fixing Function
# -------------------------------
def check_and_fix_sensors(df):
    sensor_regex = re.compile(r'(?i)^(sensor)\s*(\d+)(.*)$')
    valid_set = set(range(6))  # Valid sensor numbers: 0 to 5
    sensor_cols = [col for col in df.columns if col.lower().startswith("sensor")]
    
    present_valid = set()
    for col in sensor_cols:
        m = sensor_regex.match(col)
        if m:
            num = int(m.group(2))
            if num in valid_set:
                present_valid.add(num)
    
    missing = valid_set - present_valid
    if not missing:
        return df  # All sensors are present.
    
    missing_sensor = min(missing)
    new_mapping = {}
    for col in sensor_cols:
        m = sensor_regex.match(col)
        if m:
            num = int(m.group(2))
            if num not in valid_set:
                new_name = f"{m.group(1).lower()} {missing_sensor}{m.group(3)}"
                new_mapping[col] = new_name

    if new_mapping:
        df = df.rename(columns=new_mapping)
    
    sensor_cols = [col for col in df.columns if col.lower().startswith("sensor")]
    present_valid = set()
    for col in sensor_cols:
        m = sensor_regex.match(col)
        if m:
            num = int(m.group(2))
            if num in valid_set:
                present_valid.add(num)
                
    if valid_set.issubset(present_valid):
        return df
    else:
        print(f"After renaming, not all sensors 0-6 are present (found {present_valid}). Skipping file.")
        return None

# -------------------------------
# Function to process a single CSV file
# -------------------------------
def process_file(file_path, raw_dir, processed_dir):
    try:
        df = pd.read_csv(file_path, header=0)
        df.columns = df.columns.astype(str).str.strip()
        
        time_cols = [col for col in df.columns if col.lower().startswith("time") and col.lower() != "time 0"]
        if time_cols:
            df.drop(columns=time_cols, inplace=True)
        if "time 0" in df.columns:
            df.rename(columns={"time 0": "time"}, inplace=True)
        
        df = check_and_fix_sensors(df)
        if df is None:
            print(f"Skipping file {file_path} due to sensor requirements.")
            return None

        rel_path = os.path.relpath(file_path, raw_dir)
        rel_dir = os.path.dirname(rel_path)
        output_dir_full = os.path.join(processed_dir, rel_dir)
        os.makedirs(output_dir_full, exist_ok=True)

        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        output_csv = os.path.join(output_dir_full, f"{base_filename}.csv")
        df.to_csv(output_csv, index=False)
        return output_csv
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# -------------------------------
# Update index CSV with processed file paths and number of frames
# -------------------------------
def update_index(processed_files, processed_dir, index_filename):
    index_file = os.path.join(processed_dir, index_filename)
    records = []
    
    for f in tqdm(processed_files, desc=f"Indexing {index_filename}", unit="file"):
        try:
            df = pd.read_csv(f)
            frames = df.shape[0]
            records.append({"emg_file": f, "frames": frames})
        except Exception as e:
            print(f"Error reading {f}: {e}")

    df_index = pd.DataFrame(records)
    df_index.to_csv(index_file, index=False)
    print(f"Index updated: {index_file}")
    return len(records)

def create_index(processed_files, suffix):
    # Shuffle and split processed files into train, validation, and test sets.
    random.shuffle(processed_files)
    n = len(processed_files)
    train_end = int(n * config.train_split)
    val_end = train_end + int(n * config.val_split)
    train_files = processed_files[:train_end]
    val_files = processed_files[train_end:val_end]
    test_files = processed_files[val_end:]
    
    # Create three index files (optionally with an action suffix)
    train_count = update_index(train_files, processed_dir, f"index_train{suffix}.csv")
    val_count = update_index(val_files, processed_dir, f"index_val{suffix}.csv")
    test_count = update_index(test_files, processed_dir, f"index_test{suffix}.csv")
    
    print(f"Training index contains {train_count} files.")
    print(f"Validation index contains {val_count} files.")
    print(f"Test index contains {test_count} files.")

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    processed_dir = config.processed
    os.makedirs(processed_dir, exist_ok=True)
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

    action_filter = "_treadmill"
    # Check if the action appears as a directory in the file path.
    filter_processed_files = [f for f in processed_files if os.path.sep + action_filter + os.path.sep in f.lower()]
    suffix = f"_{action_filter}"
    print(f"After filtering, {len(filter_processed_files)} files remain for action: {action_filter}")

    create_index(filter_processed_files, action_filter)
    create_index(processed_files, "")
    # Note: The following exception about a 'NoneType' object during ProcessPoolExecutor shutdown can be safely ignored.
