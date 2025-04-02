import os
import re
import random
import numpy as np
import pandas as pd
from glob import glob
import concurrent.futures
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

# ---------------------------
# Helper functions
# ---------------------------

# Parse frequency from a column name (e.g. "(1259.259 Hz)")
FREQ_REGEX = re.compile(r'\(([\d\.]+)\s*Hz\)', re.IGNORECASE)
def parse_frequency(col_name):
    match = FREQ_REGEX.search(col_name)
    if match:
        return float(match.group(1))
    return None

# Outlier replacement: replace values more than threshold*std from the mean with the median.
def replace_outliers(data, threshold=3):
    median = np.median(data)
    mean = np.mean(data)
    std = np.std(data)
    outliers = np.abs(data - mean) > threshold * std
    data[outliers] = median
    return data

# ---------------------------
# Processing CSV files with outlier replacement, resampling & RMS for EMG
# ---------------------------
def process_csv_file(input_csv, input_base_dir, output_base_dir):
    """
    1. Reads CSV and drops the first 200 rows.
    2. Drops columns with 'Unnamed', 'IMP', or 'time'.
    3. For each remaining column:
         a. Replaces outliers,
         b. Applies min–max normalization to scale values between -1 and 1,
         c. Parses its frequency and downsamples using block averaging.
         For EMG columns, rectifies (absolute value) then computes the RMS.
    4. Splits the processed DataFrame into DS1 (all), DS2 (EMG-only), DS3 (ACC+EMG), DS4 (GYRO+EMG).
    5. Saves each DS CSV into its respective folder, preserving the relative path.
    """
    df_raw = pd.read_csv(input_csv)
    if len(df_raw) > 200:
        df_raw = df_raw.iloc[200:]
    else:
        print(f"File {input_csv} has fewer than 200 rows; skipping.")
        return

    drop_cols = [c for c in df_raw.columns if 'unnamed' in c.lower() 
                 or 'imp' in c.lower() or 'time' in c.lower()]
    df_raw = df_raw.drop(columns=drop_cols,errors='ignore')
    resampled_series_list = []
    for col in df_raw.columns:
        freq = parse_frequency(col)
        if freq is None:
            print(f"Warning: Could not parse frequency in column '{col}' of '{input_csv}'. Skipping it.")
            continue

        series = df_raw[col].dropna()
        data = series.values.astype(np.float64)
        data = replace_outliers(data, threshold=3)
        
        # Apply min-max normalization before resampling
        col_min = data.min()
        col_max = data.max()
        if col_max != col_min:
            data = 2 * (data - col_min) / (col_max - col_min) - 1
        else:
            data = np.zeros_like(data)
        
        block_size = int(round(freq / 10))
        if block_size <= 0:
            print(f"Invalid block size computed for column {col} in {input_csv}.")
            continue
        n_blocks = len(data) // block_size
        if n_blocks == 0:
            print(f"Not enough data in {input_csv} for column {col} after dropping rows.")
            continue
        
        trimmed = data[:n_blocks * block_size]
        blocks = trimmed.reshape(n_blocks, block_size)
        
        downsampled = blocks.mean(axis=1)
        
        s_down = pd.Series(downsampled, name=col)
        resampled_series_list.append(s_down)
    
    if not resampled_series_list:
        print(f"Warning: After processing columns, no valid data left for '{input_csv}'.")
        return
    
    # Concatenate all downsampled series without additional normalization
    df_down = pd.concat(resampled_series_list, axis=1)
        
    # Split into DS1 (all), DS2 (EMG-only), DS3 (ACC+EMG), DS4 (GYRO+EMG)
    ds1 = df_down.copy()
    emg_cols = [c for c in df_down.columns if 'emg' in c.lower()]
    ds2 = df_down[emg_cols].copy() if emg_cols else pd.DataFrame(index=df_down.index)
    acc_cols = [c for c in df_down.columns if ('acc' in c.lower() or 'emg' in c.lower())]
    ds3 = df_down[acc_cols].copy() if acc_cols else pd.DataFrame(index=df_down.index)
    gyro_cols = [c for c in df_down.columns if ('gyro' in c.lower() or 'emg' in c.lower())]
    ds4 = df_down[gyro_cols].copy() if gyro_cols else pd.DataFrame(index=df_down.index)
    
    rel_path = os.path.relpath(os.path.dirname(input_csv), start=input_base_dir)
    base_name = os.path.basename(input_csv)
    
    ds1_outdir = os.path.join(output_base_dir, "DS1", rel_path)
    ds2_outdir = os.path.join(output_base_dir, "DS2", rel_path)
    ds3_outdir = os.path.join(output_base_dir, "DS3", rel_path)
    ds4_outdir = os.path.join(output_base_dir, "DS4", rel_path)
    
    os.makedirs(ds1_outdir, exist_ok=True)
    os.makedirs(ds2_outdir, exist_ok=True)
    os.makedirs(ds3_outdir, exist_ok=True)
    os.makedirs(ds4_outdir, exist_ok=True)
    
    ds1.to_csv(os.path.join(ds1_outdir, base_name))
    ds2.to_csv(os.path.join(ds2_outdir, base_name))
    ds3.to_csv(os.path.join(ds3_outdir, base_name))
    ds4.to_csv(os.path.join(ds4_outdir, base_name))
    
    print(f"Processed {input_csv} -> DS1, DS2, DS3, DS4 saved. Length of DS1 csv: {len(ds1)}")

def process_all_csvs_parallel(input_base_dir, output_base_dir):
    """
    Recursively finds CSV files under input_base_dir (skipping 'camera_' subdirs)
    and processes them in parallel.
    """
    tasks = []
    print("Started finding all CSV files")
    for root, dirs, files in os.walk(input_base_dir):
        dirs[:] = [d for d in dirs if not d.lower().startswith("camera_")]
        for file in files:
            if file.lower().endswith('.csv'):
                input_csv = os.path.join(root, file)
                tasks.append(input_csv)
    
    print(f"Found {len(tasks)} CSV file(s) to process in '{input_base_dir}'.")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_csv_file, csv, input_base_dir, output_base_dir) 
                   for csv in tasks]
        for _ in tqdm(concurrent.futures.as_completed(futures),
                      total=len(futures),
                      desc="Processing CSV Files"):
            pass


# ---------------------------
# Sliding Windows Extraction (Aggregation)
# ---------------------------

def extract_sliding_windows_from_file(input_csv, lag=30, nahead=10, max_windows=20):
    """
    Reads a processed CSV file (one of the DS files), standardizes column names,
    and extracts up to `max_windows` flattened sliding windows.
    Returns a list of dictionaries.
    """
    df = pd.read_csv(input_csv, index_col=0)
    
    total_length = lag + nahead
    windows = []
    for start in range(max_windows):
        if start + total_length <= len(df):
            window_df = df.iloc[start:start+total_length].reset_index(drop=True)
            flat = {}
            for col in window_df.columns:
                for t in range(lag):
                    flat[f"{col}_lag{t}"] = window_df.loc[t, col]
                for t in range(nahead):
                    flat[f"{col}_nahead{t}"] = window_df.loc[t+lag, col]
            windows.append(flat)
        else:
            break
    return windows

def create_sliding_windows_sample(output_base_dir, figures_dir, ds_list=["DS1", "DS2", "DS3", "DS4"], lag=30, nahead=10, sample_windows=20):
    """
    For each dataset, extracts sliding windows from the first few CSV files
    and saves the first sample_windows (flattened) into a single CSV file.
    """
    for ds in ds_list:
        ds_path = os.path.join(output_base_dir, ds)
        csv_files = sorted(glob(os.path.join(ds_path, "**", "*.csv"), recursive=True))
        all_windows = []
        for csv_file in csv_files:
            windows = extract_sliding_windows_from_file(csv_file, lag, nahead, max_windows=sample_windows)
            if windows:
                all_windows.extend(windows)
            if len(all_windows) >= sample_windows:
                break
        if all_windows:
            sample = all_windows[:sample_windows]
            df_sample = pd.DataFrame(sample)
            output_dir = os.path.join(figures_dir, "datasets", ds)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "windows_sample.csv")
            df_sample.to_csv(output_file, index=False)
            print(f"Sliding windows sample saved to {output_file}")
        else:
            print(f"No sliding windows extracted for dataset {ds}")

# ---------------------------
# Index building for train/val/test splits
# ---------------------------

def build_index_for_all_datasets(output_base_dir, ds_list=["DS1", "DS2", "DS3", "DS4"], seed=42, train_ratio=0.7, val_ratio=0.15):
    ds1_files = glob(os.path.join(output_base_dir, "DS1", "**", "*.csv"), recursive=True)
    base_index = []
    for file in ds1_files:
        rel_path = os.path.relpath(file, start=output_base_dir)
        parts = rel_path.split(os.sep)
        try:
            uuid = parts[1]
        except IndexError:
            uuid = "unknown"
        base_index.append({'uuid': uuid})
    base_df = pd.DataFrame(base_index).drop_duplicates(subset="uuid")
    uuids = base_df['uuid'].tolist()
    
    random.seed(seed)
    random.shuffle(uuids)
    n = len(uuids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_uuids = set(uuids[:n_train])
    val_uuids = set(uuids[n_train:n_train+n_val])
    test_uuids = set(uuids[n_train+n_val:])
    
    indexes = {}
    for ds in ds_list:
        ds_files = glob(os.path.join(output_base_dir, ds, "**", "*.csv"), recursive=True)
        index_list = []
        for file in ds_files:
            rel_path = os.path.relpath(file, start=output_base_dir)
            parts = rel_path.split(os.sep)
            try:
                uuid = parts[1]
                action = parts[2]
            except IndexError:
                uuid = "unknown"
                action = "unknown"
            index_list.append({
                'file_path': f"./{rel_path}",
                'uuid': uuid,
                'action': action
            })
        df_index = pd.DataFrame(index_list)
        def get_split(uuid):
            if uuid in train_uuids:
                return 'train'
            elif uuid in val_uuids:
                return 'val'
            elif uuid in test_uuids:
                return 'test'
            else:
                return 'unknown'
        df_index['split'] = df_index['uuid'].apply(get_split)
        indexes[ds] = df_index
    return indexes

def save_index(index_df, output_csv):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    index_df.to_csv(output_csv, index=False)
    print(f"Index saved to: {output_csv}")

def build_train_val_test_indexes(output_base_dir, ds_list=["DS1", "DS2", "DS3", "DS4"], seed=42, train_ratio=0.7, val_ratio=0.10):
    indexes = build_index_for_all_datasets(output_base_dir, ds_list=ds_list, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)
    for ds, df_index in indexes.items():
        ds_index_path = os.path.join(output_base_dir, ds)
        os.makedirs(ds_index_path, exist_ok=True)
        for split in ['train', 'val', 'test']:
            split_df = df_index[df_index['split'] == split].drop(columns=['split'])
            save_index(split_df, os.path.join(ds_index_path, f"{split}.csv"))
    for ds in ds_list:
        ds_files = glob(os.path.join(output_base_dir, ds, "**", "*.csv"), recursive=True)
        print(f"Total CSV files in {ds}: {len(ds_files)}")

# ---------------------------
# Plotting sliding window example as PNG with subplots
# ---------------------------

def extract_one_window(input_csv, lag=30, nahead=10):
    """
    Reads a processed CSV file and extracts a single sliding window (first window)
    as a DataFrame (without flattening).
    """
    df = pd.read_csv(input_csv, index_col=0)
    total_length = lag + nahead
    if len(df) >= total_length:
        return df.iloc[:total_length]
    else:
        return None

def plot_sliding_windows(output_base_dir, figures_dir, ds_list=["DS1", "DS2", "DS3", "DS4"], lag=30, nahead=10):
    """
    For each dataset in ds_list, finds one CSV file,
    extracts a sliding window (non-flattened),
    and plots signals grouped by sensor number:
      - Group 1: Sensors 0-2
      - Group 2: Sensors 3-5
    One PNG is saved per dataset.
    """
    import re
    # Regex to extract sensor number from standardized column names (e.g., "sensor_0_emg")
    sensor_pattern = re.compile(r'sensor_(\d+)', re.IGNORECASE)
    
    for ds in ds_list:
        ds_path = os.path.join(output_base_dir, ds)
        csv_files = sorted(glob(os.path.join(ds_path, "**", "*.csv"), recursive=True))
        if not csv_files:
            print(f"No CSV files found for dataset {ds}")
            continue

        window = extract_one_window(csv_files[0], lag, nahead)
        if window is None:
            print(f"Not enough data in {csv_files[0]} to extract a window for {ds}.")
            continue

        # Group columns based on sensor number:
        # Group 1: sensors 0-2, Group 2: sensors 3-5
        group_0_2 = []
        group_3_5 = []
        for col in window.columns:
            m = sensor_pattern.search(col.lower())
            if m:
                sensor_num = int(m.group(1))
                if sensor_num <= 2:
                    group_0_2.append(col)
                elif sensor_num <= 5:
                    group_3_5.append(col)
            # Columns without a matching sensor pattern are ignored

        groups = []
        if group_0_2:
            groups.append(("Sensors 0-2", group_0_2))
        if group_3_5:
            groups.append(("Sensors 3-5", group_3_5))
        if not groups:
            print(f"No sensor data found in {csv_files[0]} for dataset {ds}.")
            continue

        n_groups = len(groups)
        fig, axes = plt.subplots(n_groups, 1, figsize=(10, 4 * n_groups))
        # Ensure axes is a list when only one subplot exists
        if n_groups == 1:
            axes = [axes]

        for ax, (title, cols) in zip(axes, groups):
            for col in cols:
                ax.plot(window.index, window[col], marker='o', linestyle='-', markersize=4, label=col)
            ax.set_title(title)
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Value")
            ax.legend()
        
        plt.tight_layout()
        output_png = os.path.join(figures_dir, "datasets", ds, f"{ds}_window_signals.png")
        os.makedirs(os.path.dirname(output_png), exist_ok=True)
        plt.savefig(output_png)
        print(f"Sliding window signals plot for {ds} saved to {output_png}")
        plt.close()


# ---------------------------
# Main processing routines
# ---------------------------

def full_processing():
    input_base_dir = "/data1/dnicho26/EMG_DATASET/data/processed"
    output_base_dir = "/data1/dnicho26/EMG_DATASET/final-data"
    figures_dir = "/data1/dnicho26/Thesis/AI-Assisted-Gait-Restoration-for-Disabled-Individuals/figures"
    
    print("Starting CSV processing...")
    process_all_csvs_parallel(input_base_dir, output_base_dir)
    
    print("Building train/val/test indexes...")
    build_train_val_test_indexes(output_base_dir, ds_list=["DS1", "DS2", "DS3", "DS4"], seed=42)
    
    print("Extracting sliding window sample across each dataset (CSV output)...")
    create_sliding_windows_sample(output_base_dir, figures_dir, ds_list=["DS1", "DS2", "DS3", "DS4"], lag=30, nahead=10, sample_windows=20)
    
    print("Plotting sliding window example (PNG with subplots)...")
    plot_sliding_windows(output_base_dir, figures_dir, ds_list=["DS1", "DS2", "DS3", "DS4"], lag=30, nahead=10)

def index_only():
    output_base_dir = "/data1/dnicho26/EMG_DATASET/final-data"
    print("Building train/val/test indexes only...")
    build_train_val_test_indexes(output_base_dir, ds_list=["DS1", "DS2", "DS3", "DS4"], seed=42)
    figures_dir = "/data1/dnicho26/Thesis/AI-Assisted-Gait-Restoration-for-Disabled-Individuals/figures"

    for ds in ["DS1", "DS2", "DS3", "DS4"]:
        ds_files = glob(os.path.join(output_base_dir, ds, "**", "*.csv"), recursive=True)
        print(f"Total CSV files in {ds}: {len(ds_files)}")

    # print("Extracting sliding window sample across each dataset (CSV output)...")
    # create_sliding_windows_sample(output_base_dir, figures_dir, ds_list=["DS1", "DS2", "DS3", "DS4"], lag=30, nahead=10, sample_windows=20)
    
    # print("Plotting sliding window example (PNG with subplots)...")
    # plot_sliding_windows(output_base_dir, figures_dir, ds_list=["DS1", "DS2", "DS3", "DS4"], lag=30, nahead=10)


if __name__ == "__main__":
    # Uncomment argparse handling if running from command line
    # parser = argparse.ArgumentParser(description="Process CSVs and/or build indexes for DS folders.")
    # parser.add_argument("--index-only", action="store_true", help="Build index only, skipping CSV processing.")
    # args = parser.parse_args()
    index_only_mode = True
    if index_only_mode:
        index_only()
    else:
        full_processing()
