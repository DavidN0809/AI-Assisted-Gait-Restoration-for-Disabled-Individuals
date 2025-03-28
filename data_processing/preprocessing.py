#!/usr/bin/env python3
import os
import re
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
from concurrent.futures import ThreadPoolExecutor

# -------------------------------
# Minimal Configuration for dataset paths and splits
# -------------------------------
class Configuration:
    def __init__(self, root, train_split=0.7, val_split=0.10, test_split=0.20):
        self.root = root
        self.raw = os.path.join(root, "data/data")        # raw CSV files
        self.processed = os.path.join(root, "data/processed-server")
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

# Update this to your base directory.
base_dir = "/data1/dnicho26/EMG_DATASET/"
config = Configuration(root=base_dir)

def find_csv_in_subdir(subdir):
    csv_files = []
    for root, dirs, files in os.walk(subdir, topdown=True):
        # Remove any directories with "camera" in their name (case insensitive)
        dirs[:] = [d for d in dirs if "camera" not in d.lower()]
        for file in files:
            if file.lower().endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

def find_csv_files(directory):
    csv_files = []
    # Get immediate subdirectories
    subdirs = [os.path.join(directory, d) for d in os.listdir(directory)
               if os.path.isdir(os.path.join(directory, d))]
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(find_csv_in_subdir, subdir) for subdir in subdirs]
        for future in futures:
            csv_files.extend(future.result())
    
    return csv_files

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
        print(f"After renaming, not all sensors 0-5 are present (found {present_valid}). Skipping file.")
        return None

# -------------------------------
# New Normalization Function
# -------------------------------
def normalize_dataframe(df, clip_threshold=3):
    """
    Normalize numeric columns (excluding 'time') using z-score normalization.
    Values are clipped to the range [-clip_threshold, clip_threshold].
    """
    for col in df.columns:
        if col.lower() == "time":
            continue  # Skip normalization for time column.
        if pd.api.types.is_numeric_dtype(df[col]):
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val != 0:
                # Compute z-score normalization
                normalized = (df[col] - mean_val) / std_val
                # Clip outliers to handle extreme values
                df[col] = normalized.clip(-clip_threshold, clip_threshold)
    return df

# -------------------------------
# Function to process a single CSV file
# -------------------------------
def process_file(file_path, raw_dir, processed_dir):
    try:
        df = pd.read_csv(file_path, header=0)
        df.columns = df.columns.astype(str).str.strip()
        
        # Remove extra time columns and rename "time 0" to "time"
        time_cols = [col for col in df.columns if col.lower().startswith("time") and col.lower() != "time 0"]
        if time_cols:
            df.drop(columns=time_cols, inplace=True)
        if "time 0" in df.columns:
            df.rename(columns={"time 0": "time"}, inplace=True)
        
        df = check_and_fix_sensors(df)
        if df is None:
            print(f"Skipping file {file_path} due to sensor requirements.")
            return None

        # Check sample rate if "time" column exists.
        if "time" in df.columns:
            # Compute time differences and use median as representative dt
            time_diffs = df["time"].diff().dropna()
            if not time_diffs.empty:
                median_dt = time_diffs.median()
                # Avoid division by zero
                if median_dt == 0:
                    print(f"File {file_path} has zero time difference. Skipping file.")
                    return None
                computed_sample_rate = 1 / median_dt
                # Allow a tolerance of 1 Hz around 1259.259 Hz
                if abs(computed_sample_rate - 1259.259) > 1:
                    print(f"Dropping file {file_path} due to sample rate mismatch: computed {computed_sample_rate:.3f} Hz")
                    return None

        # -------------------------------
        # Apply normalization per column
        # -------------------------------
        df = normalize_dataframe(df)

        # Prepare output directory and save the processed file
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
    train_count = update_index(train_files, config.processed, f"index_train{suffix}.csv")
    val_count = update_index(val_files, config.processed, f"index_val{suffix}.csv")
    test_count = update_index(test_files, config.processed, f"index_test{suffix}.csv")
    
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
    suffix = f"{action_filter}"
    print(f"After filtering, {len(filter_processed_files)} files remain for action: {action_filter}")

    create_index(filter_processed_files, action_filter)
    create_index(processed_files, "")
    # Note: The following exception about a 'NoneType' object during ProcessPoolExecutor shutdown can be safely ignored.
