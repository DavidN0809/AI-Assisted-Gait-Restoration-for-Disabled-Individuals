import os
import numpy as np
import pandas as pd
import logging
import functools
import uuid
from sklearn.preprocessing import MinMaxScaler
import scipy.signal as signal
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set this flag to True to skip processing and just build the index files.
INDEX_ONLY = True

base_dir = r"D:\UNC Charlotte Dropbox\orgs-ecgr-QuantitativeImagingandAILaboratory"
# base_dir = "/data1/dnicho26/EMG_DATASET/data"

# Define custom VERBOSE logging level (most detailed).
VERBOSE_LEVEL_NUM = 5
logging.addLevelName(VERBOSE_LEVEL_NUM, "VERBOSE")

def verbose(self, message, *args, **kwargs):
    if self.isEnabledFor(VERBOSE_LEVEL_NUM):
        self._log(VERBOSE_LEVEL_NUM, message, args, **kwargs)

logging.Logger.verbose = verbose
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def log_entry_exit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.verbose(f"Entering {func.__name__}")
        result = func(*args, **kwargs)
        logger.verbose(f"Exiting {func.__name__}")
        return result
    return wrapper

def count_frames(file):
    with open(file, 'r') as f:
        return sum(1 for line in f) - 1

def find_csv_files(base_path):
    csv_files = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files

def resample_data(df, target_fs, original_fs):
    original_time = df['time'].astype(float).values
    t_start = original_time[0]
    t_end = original_time[-1]
    original_length = len(df)
    target_length = int(round(original_length * target_fs / original_fs))
    new_time = np.linspace(t_start, t_end, target_length)
    df_resampled = pd.DataFrame({'time': new_time})
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
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*hz", sensor_name, re.IGNORECASE)
    if matches:
        freqs = [float(match) for match in matches]
        return max(freqs)
    return None

@log_entry_exit
def fix_sensor_columns(df, expected_range, log_renames=False):
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
    for col in df.columns:
        if any(k in str(col).upper() for k in ["EMG", "ACC", "GYRO"]):
            if f"SENSOR {sensor_num}" in str(col).upper():
                return True
    return False

@log_entry_exit
def check_valid_file(file):
    try:
        # Use the Python engine to help mitigate C engine memory issues.
        df = pd.read_csv(file, header=None, engine='python')
    except Exception as e:
        logger.error(f"Error reading {file}: {e}")
        return None

    df.columns = df.iloc[0]
    df = df.drop(0, axis=0).reset_index(drop=True)
    df.columns = df.columns.astype(str).str.strip()
    
    df = fix_sensor_columns(df, expected_range)
    
    for num in expected_range:
        if not sensor_present(df, num):
            logger.info(f"File '{file}' is missing sensor {num}.")
            return None
    return (file, df)

# Global cache for validated files.
valid_files_data = {}

@log_entry_exit
def process_stage(stage):
    processed_files = []
    with ThreadPoolExecutor(max_workers=threads_per_stage) as executor:
        futures = {executor.submit(process_file, file, stage): file for file in valid_files_data}
        for future in as_completed(futures):
            result = future.result()
            if result:
                processed_files.append(result)
    return stage, processed_files

@log_entry_exit
def process_file(file, stage):
    try:
        logger.debug(f"Starting processing of file: {file} for stage: {stage}")
        relative_path = os.path.relpath(file, base_dir)
        relative_dir = os.path.dirname(relative_path)
        fixed_relative_dir = os.path.join(*[part.replace(" ", "_") for part in relative_dir.split(os.sep)]) if relative_dir else ""
        output_path = os.path.join(base_dir, stage, fixed_relative_dir)
        os.makedirs(output_path, exist_ok=True)
        output_filename = os.path.join(output_path, os.path.basename(file))
        
        if os.path.exists(output_filename):
            logger.info(f"Skipping {file} for stage {stage} as it already exists at {output_filename}.")
            return output_filename

        df = valid_files_data[file].copy()
        df.columns = df.columns.astype(str).str.strip()

        time_cols_to_drop = [col for col in df.columns 
                             if str(col).strip().lower().startswith('time') and str(col).strip().lower() != 'time 0']
        if time_cols_to_drop:
            df.drop(columns=time_cols_to_drop, inplace=True)
        df.rename(columns={'time 0': 'time'}, inplace=True)
        
        df['time'] = pd.to_numeric(df['time'], errors='coerce')
        df.fillna(0, inplace=True)
        df, renames = fix_sensor_columns(df, expected_range, log_renames=True)
        
        sensor_cols = [col for col in df.columns if any(k in str(col).upper() for k in ["EMG", "ACC", "GYRO"])]
        
        if stage == "preprocessed":
            processed_df = df.copy()
        else:
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
        
        processed_df.to_csv(output_filename, index=False)
        logger.info(f"Processed and saved [{stage}]: {output_filename}")
        
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


def update_index(stage, file_list=None):
    """
    Update the index CSV for a given stage.
    If file_list is provided, use that list; otherwise, recursively scan the stage directory.
    For each file, extract FRAMES count, and derive uuid and action from the file path.
    """
    stage_dir = os.path.join(base_dir, stage)
    os.makedirs(stage_dir, exist_ok=True)
    index_file = os.path.join(stage_dir, "index.csv")
    
    # If file_list is not provided, scan the stage directory recursively.
    if file_list is None:
        stage_files = []
        for root, _, files in os.walk(stage_dir):
            for file in files:
                if file.endswith(".csv") and file != "index.csv":
                    stage_files.append(os.path.join(root, file))
    else:
        stage_files = file_list

    # Load the existing index if it exists; otherwise, create a new DataFrame.
    if os.path.exists(index_file):
        df_index = pd.read_csv(index_file)
    else:
        df_index = pd.DataFrame(columns=["file_path", "FRAMES", "uuid", "action"])

    # Ensure required columns exist.
    for col in ["FRAMES", "uuid", "action"]:
        if col not in df_index.columns:
            df_index[col] = None

    for file in stage_files:
        # Extract relative path from the stage directory.
        relative_path = os.path.relpath(file, stage_dir)
        parts = relative_path.split(os.sep)
        # Expecting at least three parts: <uuid>/<action>/<filename>
        if len(parts) >= 3:
            file_uuid = parts[0]
            file_action = parts[1]
        elif len(parts) >= 2:
            file_uuid = parts[0]
            file_action = parts[1]
        elif len(parts) >= 1:
            file_uuid = parts[0]
            file_action = ""
        else:
            file_uuid = ""
            file_action = ""
            
        frames = count_frames(file)
        
        # Check if file already exists in the index.
        if file not in df_index["file_path"].values:
            new_row = {"file_path": file, "FRAMES": frames, "uuid": file_uuid, "action": file_action}
            new_row_df = pd.DataFrame([new_row])
            df_index = pd.concat([df_index, new_row_df], ignore_index=True)
            logger.info(f"Stage: {stage} (INDEX) Added new file to index: {file}, uuid is {file_uuid}, action is {file_action}")
        else:
            idx = df_index.index[df_index["file_path"] == file][0]
            df_index.at[idx, "FRAMES"] = frames
            df_index.at[idx, "uuid"] = file_uuid
            df_index.at[idx, "action"] = file_action
            logger.info(f"Stage: {stage} (INDEX) Added new file to index: {file}, uuid is {file_uuid}, action is {file_action}")

    df_index.to_csv(index_file, index=False)
    logger.info(f"Index file saved for {stage}: {index_file}")


# Base directory where raw CSVs are stored.
csv_files = find_csv_files(base_dir)
if not csv_files:
    raise ValueError("No CSV files found.")

expected_range = set(range(0, 6))
expected_sensors = [f"sensor {i}" for i in range(0, 6)]

# Determine available cores.
total_cores = os.cpu_count() or 4
reserved_cores = 4
usable_cores = max(1, total_cores - reserved_cores)
logger.info(f"Total cores: {total_cores}, reserved: {reserved_cores}, usable: {usable_cores}")

# Define processing stages.
stages = ["preprocessed", "preprocessed_resampled", "preprocessed_butterworth"]
threads_per_stage = max(1, usable_cores // len(stages))
logger.info(f"Threads per stage: {threads_per_stage}")


if not INDEX_ONLY:
    # Define processing stages.
    stages = ["preprocessed", "preprocessed_resampled", "preprocessed_butterworth"]
    threads_per_stage = max(1, usable_cores // len(stages))
    logger.info(f"Threads per stage: {threads_per_stage}")
    # Always validate files concurrently.
    with ThreadPoolExecutor(max_workers=usable_cores) as executor:
        futures = {executor.submit(check_valid_file, file): file for file in csv_files}
        for future in as_completed(futures):
            result = future.result()
            if result:
                file, df = result
                valid_files_data[file] = df
    logger.info(f"Found {len(valid_files_data)} valid files out of {len(csv_files)} total CSVs.")

    processed_files_by_stage = {}
    # Limit the number of stage threads as well.
    with ThreadPoolExecutor(max_workers=min(len(stages), usable_cores)) as stage_executor:
        stage_futures = {stage_executor.submit(process_stage, stage): stage for stage in stages}
        for future in as_completed(stage_futures):
            stage, files = future.result()
            processed_files_by_stage[stage] = files

    # Update index for each stage using processed files.
    for stage, file_list in processed_files_by_stage.items():
        update_index(stage, file_list=file_list)
else:
    # INDEX_ONLY mode: update index using the validated files
    stages = ["preprocessed", "preprocessed_resampled", "preprocessed_butterworth"]
    validated_files = list(valid_files_data.keys())
    for stage in stages:
        update_index(stage)

