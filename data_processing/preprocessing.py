#!/usr/bin/env python3
import os
import re
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import random

# --- User Settings ---
base_dir = "/data1/dnicho26/EMG_DATASET"
raw_dir = os.path.join(base_dir, "data/data")          # where raw CSVs reside
processed_dir = os.path.join(base_dir, "processed")
os.makedirs(processed_dir, exist_ok=True)

# --- Helper Functions ---
def find_csv_files(directory):
    """Recursively find all CSV files under the given directory, skipping folders with 'camera'."""
    csv_files = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if "camera" not in d.lower()]
        for file in files:
            if file.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files

def fix_sensor_columns(df):
    """
    Automatically verify and fix sensor columns.
    Expected sensor groups: "sensor X ..." where X is in 0-5.
    
    This function:
      1. Groups sensor columns (using a regex) by their sensor number.
      2. Determines which valid sensor numbers (0–5) are missing.
      3. For each invalid sensor group (e.g. sensor 7), it renames *all* its columns
         to a missing sensor number (the smallest missing one is used first).
      4. Prints details for each renaming.
      
    If after remapping the complete set {0,1,2,3,4,5} is not present, returns None.
    """
    sensor_pattern = re.compile(r'(?i)^(sensor)\s*(\d+)(.*)$')
    valid_set = set(range(6))  # expected sensor groups: 0-5

    # Group columns by sensor number.
    sensor_groups = {}  # sensor number -> list of column names
    for col in df.columns:
        m = sensor_pattern.match(col)
        if m:
            num = int(m.group(2))
            sensor_groups.setdefault(num, []).append(col)
    
    present = set(num for num in sensor_groups if num in valid_set)
    missing = sorted(list(valid_set - present))
    
    # Find invalid sensor groups (those not in the expected range)
    invalid_groups = {num: cols for num, cols in sensor_groups.items() if num not in valid_set}
    
    # If no invalid groups and some valid sensors are missing, there's nothing to remap.
    if missing and not invalid_groups:
        print(f"Skipping file because sensors are incomplete. Found sensors: {present}")
        return None

    rename_mapping = {}
    # For each invalid group in sorted order, assign a missing sensor number if available.
    for invalid_num in sorted(invalid_groups.keys()):
        if not missing:
            break
        target = missing.pop(0)
        for col in invalid_groups[invalid_num]:
            m = sensor_pattern.match(col)
            suffix = m.group(3) if m else ""
            new_name = f"sensor {target}{suffix}"
            rename_mapping[col] = new_name
            print(f"Renaming '{col}' (sensor {invalid_num}) to '{new_name}'")
    
    # Apply the renaming
    df = df.rename(columns=rename_mapping)

    # Verify that sensor groups now are exactly valid_set.
    sensor_groups_after = set()
    for col in df.columns:
        m = sensor_pattern.match(col)
        if m:
            sensor_groups_after.add(int(m.group(2)))
    if sensor_groups_after != valid_set:
        print(f"Warning: After renaming, sensors present: {sensor_groups_after}, expected: {valid_set}.")
        return None

    return df

def process_file(file_path):
    """
    Process one CSV file:
      - Read CSV.
      - Remove all time columns except "time 0" and rename "time 0" to "time".
      - Remove sensor columns containing "IMP".
      - Verify that the resulting DataFrame has 43 columns.
      - Fix sensor columns so that sensors 0-5 exist.
      - Save the processed CSV under the processed directory preserving relative path.
    """
    try:
        df = pd.read_csv(file_path, header=0)
        # Strip extra whitespace from column names.
        df.columns = df.columns.astype(str).str.strip()

        # 1. Remove all time columns except "time 0" and rename it to "time".
        time_cols = [col for col in df.columns if col.lower().startswith("time") and col.lower() != "time 0"]
        if time_cols:
            df.drop(columns=time_cols, inplace=True)
        if "time 0" in df.columns:
            df.rename(columns={"time 0": "time"}, inplace=True)
        else:
            print(f"File {file_path} missing 'time 0' column. Skipping.")
            return None

        # 2. Remove all sensor columns containing "IMP" (case insensitive).
        imp_cols = [col for col in df.columns if "imp" in col.lower()]
        if imp_cols:
            df.drop(columns=imp_cols, inplace=True)

        # 3. Check column count.
        if df.shape[1] != 44:
            print(f"File {file_path} skipped because column count after time/IMP removal is {df.shape[1]} (expected 43).")
            return None

        # 4. Fix sensor columns so that sensors 0-5 exist.
        df = fix_sensor_columns(df)
        if df is None:
            print(f"File {file_path} skipped due to sensor column issues after fixing.")
            return None

        # Save processed CSV maintaining relative path structure.
        rel_path = os.path.relpath(file_path, raw_dir)
        out_dir = os.path.join(processed_dir, os.path.dirname(rel_path))
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, os.path.basename(file_path))
        df.to_csv(out_file, index=False)
        return out_file

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    csv_files = find_csv_files(raw_dir)
    print(f"Found {len(csv_files)} CSV files.")

    processed_files = []
    # Use ProcessPoolExecutor to process files in parallel.
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_file, file): file for file in csv_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files", unit="file"):
            result = future.result()
            if result:
                processed_files.append(result)

    print(f"Successfully processed {len(processed_files)} files.")

