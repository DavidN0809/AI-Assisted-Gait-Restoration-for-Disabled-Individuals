#!/usr/bin/env python3
import os
import re
import random
import numpy as np
import pandas as pd
from glob import glob
import concurrent.futures
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import time
import io
import contextlib
from ultralytics import YOLO

# =============================================================================
# EMG CSV Processing Functions
# =============================================================================

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
    df_raw = df_raw.drop(columns=drop_cols, errors='ignore')
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

# =============================================================================
# Sliding Window Extraction & Index Building for EMG Datasets
# =============================================================================

def extract_sliding_windows_from_file(input_csv, lag=30, nahead=10, max_windows=20):
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

def plot_sliding_windows(output_base_dir, figures_dir, ds_list=["DS1", "DS2", "DS3", "DS4"], lag=30, nahead=10):
    import re
    sensor_pattern = re.compile(r'sensor_(\d+)', re.IGNORECASE)
    
    for ds in ds_list:
        ds_path = os.path.join(output_base_dir, ds)
        csv_files = sorted(glob(os.path.join(ds_path, "**", "*.csv"), recursive=True))
        if not csv_files:
            print(f"No CSV files found for dataset {ds}")
            continue

        # Extract first sliding window from first CSV
        df = pd.read_csv(csv_files[0], index_col=0)
        total_length = lag + nahead
        if len(df) < total_length:
            print(f"Not enough data in {csv_files[0]} to extract a window for {ds}.")
            continue
        window = df.iloc[:total_length].reset_index(drop=True)

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

# =============================================================================
# Pose Extraction Functions for Camera Files
# =============================================================================

# Initialize YOLO pose model
pose_model = YOLO('yolov8n-pose.pt')

# Define keypoint indices (COCO format)
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

skeleton_connections = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6), (5, 11),
    (6, 12), (11, 13),
    (13, 15), (12, 14),
    (14, 16), (11, 12)
]

def draw_keypoints(image, keypoints, confidence_threshold=0.3):
    num_keypoints = len(keypoints) // 3
    for i in range(num_keypoints):
        x = int(keypoints[i * 3])
        y = int(keypoints[i * 3 + 1])
        conf = keypoints[i * 3 + 2]
        if conf > confidence_threshold:
            cv2.circle(image, (x, y), 4, (0, 255, 0), -1)

def draw_skeleton(image, keypoints, skeleton_connections, confidence_threshold=0.3):
    for connection in skeleton_connections:
        idx1, idx2 = connection
        x1, y1, c1 = keypoints[idx1 * 3], keypoints[idx1 * 3 + 1], keypoints[idx1 * 3 + 2]
        x2, y2, c2 = keypoints[idx2 * 3], keypoints[idx2 * 3 + 1], keypoints[idx2 * 3 + 2]
        if c1 > confidence_threshold and c2 > confidence_threshold:
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
            cv2.line(image, pt1, pt2, (255, 0, 0), 2)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ab = a - b
    cb = c - b
    dot_product = np.dot(ab, cb)
    norm_ab = np.linalg.norm(ab)
    norm_cb = np.linalg.norm(cb)
    if norm_ab == 0 or norm_cb == 0:
        return 0.0
    cosine_angle = dot_product / (norm_ab * norm_cb)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle_rad = np.arccos(cosine_angle)
    return np.degrees(angle_rad)

def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def process_frame_split(frame):
    with contextlib.redirect_stdout(io.StringIO()):
        results = pose_model(frame, verbose=False)
    for result in results:
        keypoints = result.keypoints.data.cpu().numpy().flatten()
        def get_point(index):
            return (keypoints[index * 3], keypoints[index * 3 + 1])
        points = {
            'left_hip': get_point(LEFT_HIP),
            'right_hip': get_point(RIGHT_HIP),
            'left_knee': get_point(LEFT_KNEE),
            'right_knee': get_point(RIGHT_KNEE),
            'left_ankle': get_point(LEFT_ANKLE),
            'right_ankle': get_point(RIGHT_ANKLE),
            'left_shoulder': get_point(LEFT_SHOULDER),
            'right_shoulder': get_point(RIGHT_SHOULDER),
            'left_elbow': get_point(LEFT_ELBOW),
            'right_elbow': get_point(RIGHT_ELBOW),
            'left_wrist': get_point(LEFT_WRIST),
            'right_wrist': get_point(RIGHT_WRIST),
            'nose': get_point(NOSE)
        }
        left_knee_angle = calculate_angle(points['left_ankle'], points['left_knee'], points['left_hip'])
        right_knee_angle = calculate_angle(points['right_ankle'], points['right_knee'], points['right_hip'])
        left_hip_angle = calculate_angle(points['left_knee'], points['left_hip'], points['left_shoulder'])
        right_hip_angle = calculate_angle(points['right_knee'], points['right_hip'], points['right_shoulder'])
        left_elbow_angle = calculate_angle(points['left_wrist'], points['left_elbow'], points['left_shoulder'])
        right_elbow_angle = calculate_angle(points['right_wrist'], points['right_elbow'], points['right_shoulder'])
        pelvic_mid = midpoint(points['left_hip'], points['right_hip'])
        pelvic_tilt_angle = calculate_angle(points['left_hip'], pelvic_mid, points['right_hip'])
        shoulder_center = midpoint(points['left_shoulder'], points['right_shoulder'])
        trunk_lean_angle = calculate_angle(pelvic_mid, shoulder_center, points['nose'])
        return {
            'left_knee': left_knee_angle,
            'right_knee': right_knee_angle,
            'left_hip': left_hip_angle,
            'right_hip': right_hip_angle,
            'left_elbow': left_elbow_angle,
            'right_elbow': right_elbow_angle,
            'pelvic_tilt': pelvic_tilt_angle,
            'trunk_lean': trunk_lean_angle,
            'knee_symmetry': abs(left_knee_angle - right_knee_angle),
            'hip_symmetry': abs(left_hip_angle - right_hip_angle),
            'elbow_symmetry': abs(left_elbow_angle - right_elbow_angle)
        }
    return None

def process_video_folder_split(folder_path):
    files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.npy')],
        key=lambda f: extract_timestamp(f) if extract_timestamp(f) is not None else float('inf')
    )
    data = []
    for file in files:
        timestamp = extract_timestamp(file)
        if timestamp is None:
            continue
        frame = np.load(os.path.join(folder_path, file)).astype(np.uint8)
        if frame.shape[-1] != 3:
            continue
        angles = process_frame_split(frame)
        if angles:
            angles['time'] = timestamp
            data.append(angles)
    return pd.DataFrame(data)

def extract_timestamp(filename):
    try:
        base = filename.split('_')[1]
        seconds = base.split('.')[0]
        milliseconds = base.split('.')[1]
        return float(f"{seconds}.{milliseconds}")
    except (IndexError, ValueError):
        return None

def resample_split(df, target_fs=10):
    if len(df) < 2:
        return df
    start_time = df['time'].iloc[0]
    end_time = df['time'].iloc[-1]
    duration = end_time - start_time
    target_len = int(duration * target_fs)
    if target_len < 2:
        return df
    new_time = np.linspace(start_time, end_time, target_len)
    resampled = {'time': new_time}
    for column in df.columns:
        if column != 'time':
            resampled[column] = np.interp(new_time, df['time'], df[column])
    return pd.DataFrame(resampled)

# =============================================================================
# Processing Camera Files to Build New Datasets
# =============================================================================

def process_camera_folder(camera_input_dir, target_fs=10):
    """
    Process a single camera folder containing npy files.
    Returns a DataFrame with pose features (angles) over time.
    """
    df_pose = process_video_folder_split(camera_input_dir)
    if not df_pose.empty:
        df_resampled = resample_split(df_pose, target_fs=target_fs)
        return df_resampled
    return pd.DataFrame()

def process_all_cameras(camera_base_dir, output_base_dir, target_fs=10):
    """
    Process all camera directories under camera_base_dir.
    Saves processed CSV files in output_base_dir using folder names.
    """
    for cam_name in sorted(os.listdir(camera_base_dir)):
        cam_dir = os.path.join(camera_base_dir, cam_name)
        if os.path.isdir(cam_dir):
            print(f"Processing camera files for {cam_name}...")
            df_camera = process_camera_folder(cam_dir, target_fs=target_fs)
            if not df_camera.empty:
                out_dir = os.path.join(output_base_dir, f"{cam_name}_camera")
                os.makedirs(out_dir, exist_ok=True)
                out_file = os.path.join(out_dir, f"{cam_name}_camera.csv")
                df_camera.to_csv(out_file, index=False)
                print(f"Processed camera data for {cam_name} saved to {out_file}")
            else:
                print(f"No valid camera data found for {cam_name}")

# =============================================================================
# Full Processing Routine (EMG + Camera)
# =============================================================================

def full_processing_with_camera():
    # EMG processing paths
    emg_input_base_dir = "/data1/dnicho26/EMG_DATASET/data/processed"
    emg_output_base_dir = "/data1/dnicho26/EMG_DATASET/final-data-camera"
    figures_dir = "/data1/dnicho26/Thesis/AI-Assisted-Gait-Restoration-for-Disabled-Individuals/figures"
    
    # Process EMG CSVs
    print("Starting EMG CSV processing...")
    process_all_csvs_parallel(emg_input_base_dir, emg_output_base_dir)
    print("Building train/val/test indexes for EMG datasets...")
    build_train_val_test_indexes(emg_output_base_dir, ds_list=["DS1", "DS2", "DS3", "DS4"], seed=42)
    
    # Camera processing paths
    camera_base_dir = "/data1/dnicho26/EMG_DATASET/data/camera_files"  # Adjust to your actual camera files location
    camera_output_dir = "/data1/dnicho26/EMG_DATASET/final-data-camera"
    os.makedirs(camera_output_dir, exist_ok=True)
    
    print("Starting camera files processing...")
    process_all_cameras(camera_base_dir, camera_output_dir, target_fs=10)
    print("Building train/val/test indexes for camera datasets...")
    build_train_val_test_indexes(camera_output_dir, ds_list=[f"{name}_camera" for name in sorted(os.listdir(camera_base_dir)) if os.path.isdir(os.path.join(camera_base_dir, name))], seed=42)
    
    print("Full processing (EMG + camera) completed.")

if __name__ == "__main__":
    full_processing_with_camera()
