import os
import numpy as np
import pandas as pd
import logging
import functools
from sklearn.preprocessing import MinMaxScaler
import scipy.signal as signal
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define custom VERBOSE logging level (most detailed).
VERBOSE_LEVEL_NUM = 5
logging.addLevelName(VERBOSE_LEVEL_NUM, "VERBOSE")

def verbose(self, message, *args, **kwargs):
    if self.isEnabledFor(VERBOSE_LEVEL_NUM):
        self._log(VERBOSE_LEVEL_NUM, message, args, **kwargs)

# Attach the verbose method to the Logger class.
logging.Logger.verbose = verbose

# Set up logging with default level DEBUG.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

# Create a global logger instance.
logger = logging.getLogger(__name__)

# Decorator for logging function entry and exit at verbose level.
def log_entry_exit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.verbose(f"Entering {func.__name__}")
        result = func(*args, **kwargs)
        logger.verbose(f"Exiting {func.__name__}")
        return result
    return wrapper

def find_csv_files(base_path):
    """Recursively finds CSV files in the directory structure."""
    csv_files = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files

def resample_data(df, target_fs, original_fs):
    # Use the 'time' column for resampling.
    original_time = df['time'].astype(float).values
    t_start = original_time[0]
    t_end = original_time[-1]
    original_length = len(df)
    target_length = int(round(original_length * target_fs / original_fs))
    
    # Create a new uniformly spaced time vector and new DataFrame with it.
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
            logger.warning(f"Skipping column '{col}' during resampling: {e}")
            df_resampled[col] = df[col]
    return df_resampled

def safe_filtfilt(b, a, data):
    padlen = 3 * (max(len(a), len(b)) - 1)
    if len(data) <= padlen:
        logger.warning("Data length too short for filtfilt. Skipping filtering for this signal.")
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

@log_entry_exit
def fix_sensor_columns(df, expected_range, log_renames=False):
    """
    Renames sensor columns so that sensors outside the expected range are re-assigned
    to a missing sensor slot consistently.
    
    Returns:
      - If log_renames is False (default): the modified DataFrame.
      - If log_renames is True: a tuple (DataFrame, renames) where renames is a list of tuples:
            (old_sensor, new_sensor, [list of affected columns])
    """
    renames = []
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
            if log_renames:
                renames.append((extra_sensor, new_sensor, cols))
            for col in cols:
                new_col = re.sub(r"(sensor\s*)\d+", r"\g<1>" + str(new_sensor), col, flags=re.IGNORECASE)
                df.rename(columns={col: new_col}, inplace=True)
        else:
            if log_renames:
                for col in cols:
                    renames.append((extra_sensor, None, [col]))
    if log_renames:
        return df, renames
    return df

def sensor_present(df, sensor_num):
    """Return True if any column (among EMG, ACC, or GYRO) contains 'sensor {sensor_num}'."""
    for col in df.columns:
        if any(k in str(col).upper() for k in ["EMG", "ACC", "GYRO"]):
            if f"SENSOR {sensor_num}" in str(col).upper():
                return True
    return False

@log_entry_exit
def check_valid_file(file):
    """
    Reads and validates a CSV file.
    Returns a tuple (file, df) if valid; otherwise returns None.
    """
    try:
        df = pd.read_csv(file, header=None, low_memory=False)
    except Exception as e:
        logger.error(f"Error reading {file}: {e}")
        return None

    # Use the first row as header and drop it from data.
    df.columns = df.iloc[0]
    df = df.drop(0, axis=0).reset_index(drop=True)
    # Convert all column names to strings and strip whitespace.
    df.columns = df.columns.astype(str).str.strip()
    
    # Fix sensor columns (without verbose logging during validation).
    df = fix_sensor_columns(df, expected_range)
    
    # Check that every expected sensor (0–5) is present.
    for num in expected_range:
        if not sensor_present(df, num):
            logger.info(f"File '{file}' is missing sensor {num}.")
            return None
    return (file, df)

# Global cache: mapping file path to its validated DataFrame.
valid_files_data = {}

@log_entry_exit
def process_stage(stage):
    processed_files = []
    # Iterate over the cached valid files.
    with ThreadPoolExecutor(max_workers=threads_per_stage) as executor:
        futures = {executor.submit(process_file, file, stage): file for file in valid_files_data}
        for future in as_completed(futures):
            result = future.result()
            if result:
                processed_files.append(result)
    return stage, processed_files

@log_entry_exit
def process_file(file, stage):
    """
    Process a single file based on the chosen stage:
      - stage "preprocessed": sensor columns fixed only.
      - stage "preprocessed_resampled": sensor fix + resampling.
      - stage "preprocessed_butterworth": sensor fix + resampling + filtering.
    
    Uses the cached DataFrame (copied so the original remains unchanged).
    """
    try:
        logger.debug(f"Starting processing of file: {file} for stage: {stage}")
        # Use the cached DataFrame and work on a copy.
        df = valid_files_data[file].copy()

        # Ensure column names are strings (in case they changed in cache).
        df.columns = df.columns.astype(str).str.strip()

        # Remove extra time columns (e.g., "time 1", "time 2", etc.)
        time_cols_to_drop = [col for col in df.columns 
                             if str(col).strip().lower().startswith('time') and str(col).strip().lower() != 'time 0']
        if time_cols_to_drop:
            df.drop(columns=time_cols_to_drop, inplace=True)
        # Rename "time 0" to "time"
        df.rename(columns={'time 0': 'time'}, inplace=True)
        
        df['time'] = pd.to_numeric(df['time'], errors='coerce')
        df.fillna(0, inplace=True)
        # Fix sensor columns and collect renaming info.
        df, renames = fix_sensor_columns(df, expected_range, log_renames=True)
        
        # Identify sensor columns.
        sensor_cols = [col for col in df.columns if any(k in str(col).upper() for k in ["EMG", "ACC", "GYRO"])]
        
        # Stage-dependent processing.
        if stage == "preprocessed":
            processed_df = df.copy()
        else:
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

            processed_df = resample_data(df, target_fs=original_fs, original_fs=original_fs)
            
            if stage == "preprocessed_butterworth":
                processed_df = apply_filter(processed_df, sensor_cols, order=4, cutoff=[10], fs=original_fs, filter_band='low')
                processed_df = apply_notch(processed_df, sensor_cols, notch_freq=60, fs=original_fs, Q=30)
                processed_df.interpolate(method='linear', inplace=True)
            
            scaler = MinMaxScaler()
            processed_df = pd.DataFrame(scaler.fit_transform(processed_df), columns=processed_df.columns)
        
        # Determine output subfolder structure based on relative path.
        relative_path = os.path.relpath(file, base_dir)
        relative_dir = os.path.dirname(relative_path)
        if relative_dir:
            fixed_relative_dir = os.path.join(*[part.replace(" ", "_") for part in relative_dir.split(os.sep)])
        else:
            fixed_relative_dir = ""
        output_path = os.path.join(base_dir, stage, fixed_relative_dir)
        os.makedirs(output_path, exist_ok=True)
        output_filename = os.path.join(output_path, os.path.basename(file))
        
        processed_df.to_csv(output_filename, index=False)
        logger.info(f"Processed and saved [{stage}]: {output_filename}")
        
        # Log renaming info as concise verbose output.
        for old_sensor, new_sensor, cols in renames:
            if new_sensor is not None:
                logger.verbose(f"for {relative_path} renamed sensor {old_sensor} to sensor {new_sensor}")
            else:
                logger.verbose(f"for {relative_path} sensor {old_sensor} had no available slot for renaming")
        
        logger.debug(f"Finished processing of file: {file} for stage: {stage}")
        return output_filename
    except Exception as e:
        logger.error(f"Error processing {file} for stage {stage}: {e}")
        return None

# Base directory where raw CSVs are stored.
base_dir = r"D:\UNC Charlotte Dropbox\orgs-ecgr-QuantitativeImagingandAILaboratory"
csv_files = find_csv_files(base_dir)
if not csv_files:
    raise ValueError("No CSV files found.")

# Expected sensors: numbered from 0 to 5.
expected_range = set(range(0, 6))
expected_sensors = [f"sensor {i}" for i in range(0, 6)]

# Validate files concurrently and cache the valid data.
with ThreadPoolExecutor() as executor:
    futures = {executor.submit(check_valid_file, file): file for file in csv_files}
    for future in as_completed(futures):
        result = future.result()
        if result:
            file, df = result
            valid_files_data[file] = df
logger.info(f"Found {len(valid_files_data)} valid files out of {len(csv_files)} total CSVs.")

# Define stages.
stages = ["preprocessed", "preprocessed_resampled", "preprocessed_butterworth"]

# Calculate available cores.
reserved_cores = 2
total_cores = os.cpu_count() or 4  # default to 4 if unable to detect
usable_cores = total_cores - reserved_cores

# Option 1: Allocate cores equally among stages.
threads_per_stage = max(1, usable_cores // len(stages))
# Option 2: You can set threads_per_stage to a fixed number, e.g., 2

print(f"Total cores: {total_cores}, reserved: {reserved_cores}, threads per stage: {threads_per_stage}")

# Run each stage concurrently.
processed_files_by_stage = {}
with ThreadPoolExecutor(max_workers=len(stages)) as stage_executor:
    stage_futures = {stage_executor.submit(process_stage, stage): stage for stage in stages}
    for future in as_completed(stage_futures):
         stage, files = future.result()
         processed_files_by_stage[stage] = files

# Optionally, create an index CSV for each stage.
for stage, file_list in processed_files_by_stage.items():
    index_file = os.path.join(stage, "index.csv")
    df_index = pd.DataFrame({"file_path": file_list})
    df_index.to_csv(index_file, index=False)
    logger.info(f"Index file saved for {stage}: {index_file}")
