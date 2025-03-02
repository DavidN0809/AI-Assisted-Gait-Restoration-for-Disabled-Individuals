import os
import re
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

def last_valid_index(series):
    """
    Returns the last index where the series has a nonzero value.
    If all values are zero, returns None.
    """
    # Convert to numeric (if needed) and fill non-numeric with 0
    series = pd.to_numeric(series, errors='coerce').fillna(0)
    nonzero_indices = series[series != 0].index
    return nonzero_indices[-1] if len(nonzero_indices) > 0 else None

def extract_identifier(file_path):
    """
    Extracts a UUID and Action identifier from the file path.
    Assumes the structure is: .../<UUID>/<Action>/filename.csv
    Returns (UUID, Action); if not, returns (None, None).
    """
    parts = file_path.split(os.sep)
    if len(parts) >= 3:
        return parts[-3], parts[-2]
    return None, None

def check_empty_camera_dir(full_dir):
    """
    Checks if a given camera folder is empty.
    If empty, extracts UUID and Action from the folder structure.
    Returns a dictionary if empty; otherwise, returns None.
    """
    try:
        if not os.listdir(full_dir):  # Directory is empty
            parts = full_dir.split(os.sep)
            if len(parts) >= 3:
                # Assume structure: .../UUID/Action/camera
                uuid = parts[-3]
                action = parts[-2]
                return {"UUID": uuid, "Action": action}
    except Exception as e:
        print(f"Error accessing {full_dir}: {e}")
    return None

def check_empty_camera_folders(base_path):
    """
    Walks through the directory structure and returns a DataFrame of UUIDs and Actions
    for any directory whose name contains 'camera' (case-insensitive) that is empty.
    """
    camera_dirs = []
    for root, dirs, files in os.walk(base_path):
        for d in dirs:
            if "camera" in d.lower():
                full_dir = os.path.join(root, d)
                camera_dirs.append(full_dir)
    
    empty_camera_entries = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(check_empty_camera_dir, d): d for d in camera_dirs}
        for future in as_completed(futures):
            result = future.result()
            if result:
                empty_camera_entries.append(result)
    return pd.DataFrame(empty_camera_entries)

def process_csv_file(file_path):
    """
    Processes a single CSV file:
      - Reads the CSV.
      - Grabs all columns containing 'EMG' (case-insensitive).
      - Flags the file if there are not exactly 6 such columns.
      - Additionally, if any column name contains a pattern like "sensor <num>",
        checks if the sensor number is outside the expected range (0–5).
    
    Returns a tuple: (missing_entry, out_of_range_entry) where each is a dictionary
    if an issue is found, otherwise None.
    """
    uuid, action = extract_identifier(file_path)
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None

    missing_entry = None
    out_of_range_entry = None

    # Grab all columns that include 'EMG' (case-insensitive)
    emg_columns = [col for col in df.columns if 'EMG' in str(col).upper()]
    
    # Check if there are exactly 6 EMG sensor columns.
    if len(emg_columns) != 6:
        missing_entry = {
            "UUID": uuid,
            "Action": action,
            "Found EMG Columns": len(emg_columns),
            "Expected": 6
        }
    
    # Additionally, check for sensor numbers outside the expected range (0–5)
    actual_sensor_numbers = {}
    for col in emg_columns:
        match = re.search(r'sensor\s*(\d+)', col, re.IGNORECASE)
        if match:
            sensor_num = int(match.group(1))
            actual_sensor_numbers[sensor_num] = col
    
    out_of_range = {s: col for s, col in actual_sensor_numbers.items() if s not in range(6)}
    if out_of_range:
        out_str = ", ".join(f"{s} ({col})" for s, col in out_of_range.items())
        out_of_range_entry = {
            "UUID": uuid,
            "Action": action,
            "Out of Range": out_str
        }
    return missing_entry, out_of_range_entry

def check_csv_sensors(base_path):
    """
    Walks through CSV files under base_path and checks the EMG sensor columns.
    
    Two checks:
      1. If a CSV file does not include exactly 6 EMG sensor columns, it is flagged.
      2. If a CSV file has any sensor column with a number outside the range 0–5, it is flagged.
    
    Returns two DataFrames:
      - One for files missing the expected number of sensor columns.
      - One for files with out‑of‑range sensor numbers.
    """
    csv_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    missing_entries = []
    out_of_range_entries = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_csv_file, file_path): file_path for file_path in csv_files}
        for future in as_completed(futures):
            missing_entry, out_of_range_entry = future.result()
            if missing_entry:
                missing_entries.append(missing_entry)
            if out_of_range_entry:
                out_of_range_entries.append(out_of_range_entry)
    missing_df = pd.DataFrame(missing_entries)
    out_of_range_df = pd.DataFrame(out_of_range_entries)
    return missing_df, out_of_range_df

def main(base_directory):
    # Check for empty camera folders.
    empty_camera_df = check_empty_camera_folders(base_directory)
    # Check CSV files for missing sensors and out-of-range sensor numbers.
    missing_df, out_of_range_df = check_csv_sensors(base_directory)
    
    print("=== Empty Camera Folders ===")
    if empty_camera_df.empty:
        print("None")
    else:
        print(empty_camera_df.to_string(index=False))
    
    print("\n=== CSV Files Missing Sensors (Not 6 EMG sensor columns present) ===")
    if missing_df.empty:
        print("None")
    else:
        print(missing_df.to_string(index=False))
    
    # print("\n=== CSV Files with Out-of-Range Sensors (Sensor numbers not in 0–5) ===")
    # if out_of_range_df.empty:
    #     print("None")
    # else:
    #     print(out_of_range_df.to_string(index=False))

# Set your base directory below.
base_directory = r"D:\UNC Charlotte Dropbox\orgs-ecgr-QuantitativeImagingandAILaboratory"
main(base_directory)
