import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import scipy.signal as signal
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def find_csv_files(base_path):
    """Recursively finds CSV files in the directory structure."""
    csv_files = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files

def resample_data(df, target_fs, original_fs):
    # Use the 'time' column (after renaming) for resampling.
    original_time = df['time'].astype(float).values
    t_start = original_time[0]
    t_end = original_time[-1]
    original_length = len(df)
    target_length = int(round(original_length * target_fs / original_fs))
    
    # Create a new uniformly spaced time vector and start new DataFrame with it.
    new_time = np.linspace(t_start, t_end, target_length)
    df_resampled = pd.DataFrame({'time': new_time})
    
    # Resample every other column.
    for col in df.columns:
        if col == 'time':
            continue
        try:
            data = df[col].astype(float).fillna(0).values
            resampled_data = signal.resample(data, target_length)
            df_resampled[col] = resampled_data
        except Exception as e:
            logging.warning(f"Skipping column '{col}' during resampling: {e}")
            df_resampled[col] = df[col]
    return df_resampled

def safe_filtfilt(b, a, data):
    padlen = 3 * (max(len(a), len(b)) - 1)
    if len(data) <= padlen:
        logging.warning("Data length too short for filtfilt. Skipping filtering for this signal.")
        return data
    return signal.filtfilt(b, a, data)

def butterworth_filter(data, order, cutoff, fs, filter_band='low'):
    nyquist = 0.5 * fs
    normalized_cutoff = np.array(cutoff) / nyquist
    b, a = signal.butter(order, normalized_cutoff, btype=filter_band)
    return safe_filtfilt(b, a, data)

def apply_filter(df, columns, order, cutoff, fs, filter_band='low'):
    """Apply a Butterworth filter to the specified columns."""
    for col in columns:
        data = df[col].astype(float).fillna(0).values
        df[col] = butterworth_filter(data, order, cutoff, fs, filter_band)
    return df

def apply_notch(df, columns, notch_freq=60, fs=1000, Q=30):
    nyquist = 0.5 * fs
    normalized_frequency = notch_freq / nyquist
    b, a = signal.iirnotch(normalized_frequency, Q)
    for col in columns:
        data = df[col].astype(float).fillna(0).values
        df[col] = safe_filtfilt(b, a, data)
    return df

def extract_fs(sensor_name):
    """
    Extracts the sampling frequency from a sensor column name.
    Expects a pattern like '1259.259 Hz' (case insensitive).
    """
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*hz", sensor_name, re.IGNORECASE)
    if matches:
        freqs = [float(match) for match in matches]
        return max(freqs)
    return None

def fix_sensor_columns(df, expected_range):
    """
    Renames sensor columns so that sensors outside the expected range are re-assigned
    to a missing sensor slot consistently.
    """
    extra_sensors = {}
    for col in df.columns:
        if any(k in str(col).upper() for k in ["EMG", "ACC", "GYRO"]):
            m = re.search(r"sensor\s*(\d+)", str(col), re.IGNORECASE)
            if m:
                sensor_num = int(m.group(1))
                if sensor_num not in expected_range:
                    extra_sensors.setdefault(sensor_num, []).append(col)
    
    present = set()
    for col in df.columns:
        if any(k in str(col).upper() for k in ["EMG", "ACC", "GYRO"]):
            m = re.search(r"sensor\s*(\d+)", str(col), re.IGNORECASE)
            if m:
                present.add(int(m.group(1)))
    missing = sorted(list(expected_range - present))
    
    for extra_sensor, cols in extra_sensors.items():
        if missing:
            new_sensor = missing.pop(0)
            for col in cols:
                new_col = re.sub(r"(sensor\s*)\d+", r"\g<1>" + str(new_sensor), col, flags=re.IGNORECASE)
                logging.info(f"Renaming sensor in column '{col}': {extra_sensor} -> {new_sensor}")
                df.rename(columns={col: new_col}, inplace=True)
        else:
            for col in cols:
                logging.info(f"No missing sensor slot available for column '{col}' with sensor {extra_sensor}")
    return df

def sensor_present(df, sensor_num):
    """Return True if any column (among EMG, ACC, or GYRO) contains 'sensor {sensor_num}'."""
    for col in df.columns:
        if any(k in str(col).upper() for k in ["EMG", "ACC", "GYRO"]):
            if f"SENSOR {sensor_num}" in str(col).upper():
                return True
    return False

# Base directory where raw CSVs are stored.
base_dir = r"D:\UNC Charlotte Dropbox\orgs-ecgr-QuantitativeImagingandAILaboratory"
csv_files = find_csv_files(base_dir)
if not csv_files:
    raise ValueError("No CSV files found.")

# Expected sensors: numbered from 0 to 5.
expected_range = set(range(0, 6))
expected_sensors = [f"sensor {i}" for i in range(0, 6)]

def check_valid_file(file):
    try:
        df = pd.read_csv(file, header=None, low_memory=False)
    except Exception as e:
        logging.error(f"Error reading {file}: {e}")
        return None

    # Use the first row as header.
    df.columns = df.iloc[0]
    df = df.drop(0, axis=0).reset_index(drop=True)
    df = fix_sensor_columns(df, expected_range)

    # Check that every expected sensor (0–5) is present.
    for num in expected_range:
        if not sensor_present(df, num):
            logging.info(f"File '{file}' is missing sensor {num}.")
            return None
    return file

# Validate files concurrently.
valid_files = []
with ThreadPoolExecutor() as executor:
    futures = {executor.submit(check_valid_file, file): file for file in csv_files}
    for future in as_completed(futures):
        result = future.result()
        if result:
            valid_files.append(result)
logging.info(f"Found {len(valid_files)} valid files out of {len(csv_files)} total CSVs.")

# Create a directory for preprocessed files.
preprocessed_dir = os.path.join(base_dir, "preprocessed")
os.makedirs(preprocessed_dir, exist_ok=True)
preprocessed_file_list = []

def process_file(file):
    try:
        df = pd.read_csv(file, header=0, low_memory=False)
        df.columns = df.columns.str.strip()

        # Remove extra time columns (e.g., "time 1", "time 2", etc.)
        time_cols_to_drop = [col for col in df.columns 
                             if col.strip().lower().startswith('time') and col.strip().lower() != 'time 0']
        if time_cols_to_drop:
            df.drop(columns=time_cols_to_drop, inplace=True)
        # Rename "time 0" to "time"
        df.rename(columns={'time 0': 'time'}, inplace=True)
        
        df['time'] = pd.to_numeric(df['time'], errors='coerce')
        df.fillna(0, inplace=True)
        df = fix_sensor_columns(df, expected_range)
        
        # Identify sensor columns.
        sensor_cols = [col for col in df.columns if any(k in str(col).upper() for k in ["EMG", "ACC", "GYRO"])]
        
        # Determine original sampling frequency.
        freqs = []
        for col in sensor_cols:
            fs_extracted = extract_fs(str(col))
            if fs_extracted:
                freqs.append(fs_extracted)
        if freqs:
            original_fs = min(freqs)
        else:
            time_diffs = np.diff(df['time'].dropna().astype(float))
            original_fs = 1 / np.median(time_diffs) if len(time_diffs) > 0 else 1000
        
        # Resample data using the original sampling frequency.
        df = resample_data(df, target_fs=original_fs, original_fs=original_fs)
        
        # Apply Butterworth low-pass filter and notch filter.
        df = apply_filter(df, sensor_cols, order=4, cutoff=[10], fs=original_fs, filter_band='low')
        df = apply_notch(df, sensor_cols, notch_freq=60, fs=original_fs, Q=30)
        df.interpolate(method='linear', inplace=True)
        
        # Scale ALL columns using MinMaxScaler.
        scaler = MinMaxScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        
        # Determine output subfolder structure based on relative path.
        relative_path = os.path.relpath(file, base_dir)
        output_path = os.path.join(preprocessed_dir, os.path.dirname(relative_path))
        os.makedirs(output_path, exist_ok=True)
        output_filename = os.path.join(output_path, os.path.basename(file))
        
        df.to_csv(output_filename, index=False)
        logging.info(f"Processed and saved: {output_filename}")
        return output_filename
    except Exception as e:
        logging.error(f"Error processing {file}: {e}")
        return None

# Process files concurrently.
with ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_file, file): file for file in valid_files}
    for future in as_completed(futures):
        result = future.result()
        if result:
            preprocessed_file_list.append(result)

# Create an index CSV file at the top level of the preprocessed directory.
index_file = os.path.join(preprocessed_dir, "index.csv")
df_index = pd.DataFrame({"file_path": preprocessed_file_list})
df_index.to_csv(index_file, index=False)
logging.info(f"Index file saved: {index_file}")
