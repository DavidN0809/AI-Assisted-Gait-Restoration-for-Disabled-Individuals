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

########################################
# EMG Processing Functions
########################################

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
    Process an EMG CSV:
      1. Read CSV and drop the first 200 rows plus unneeded columns.
      2. For each remaining column, apply outlier replacement, min–max normalization (to [-1,1]),
         and downsample using block averaging.
      3. Split the resulting DataFrame into 4 parts:
            DS1: All channels,
            DS2: EMG-only,
            DS3: ACC+EMG,
            DS4: GYRO+EMG.
      4. Save each split CSV under output_base_dir/<DS#>/ with the same filename.
         (output_base_dir is set per (uuid, action)).
    """
    try:
        df_raw = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Error reading {input_csv}: {e}")
        return

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
        
        # Normalize to [-1,1]
        col_min = data.min()
        col_max = data.max()
        if col_max != col_min:
            data = 2 * (data - col_min) / (col_max - col_min) - 1
        else:
            data = np.zeros_like(data)
        
        block_size = int(round(freq / 10))
        if block_size <= 0:
            print(f"Invalid block size for column {col} in {input_csv}.")
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
        print(f"Warning: No valid columns in {input_csv}.")
        return
    
    df_down = pd.concat(resampled_series_list, axis=1)
    
    # Create four splits:
    ds1 = df_down.copy()
    emg_cols = [c for c in df_down.columns if 'emg' in c.lower()]
    ds2 = df_down[emg_cols].copy() if emg_cols else pd.DataFrame(index=df_down.index)
    acc_cols = [c for c in df_down.columns if ('acc' in c.lower() or 'emg' in c.lower())]
    ds3 = df_down[acc_cols].copy() if acc_cols else pd.DataFrame(index=df_down.index)
    gyro_cols = [c for c in df_down.columns if ('gyro' in c.lower() or 'emg' in c.lower())]
    ds4 = df_down[gyro_cols].copy() if gyro_cols else pd.DataFrame(index=df_down.index)
    
    base_name = os.path.basename(input_csv)
    
    # Save each DS in output_base_dir/<DS#>/
    for ds_label, df in zip(["DS1", "DS2", "DS3", "DS4"], [ds1, ds2, ds3, ds4]):
        out_dir = os.path.join(output_base_dir, ds_label)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, base_name)
        df.to_csv(out_file, index=False)
        print(f"Saved {ds_label} for {input_csv} to {out_file}")

########################################
# Pose Extraction (Camera) Functions
########################################

# Initialize YOLO pose model (ensure the model file is available)
pose_model = YOLO('yolo11n-pose.pt')

def extract_timestamp(filename):
    try:
        # Expecting format like "frame_12345.678.npy"
        base = filename.split('_')[1]
        seconds = base.split('.')[0]
        milliseconds = base.split('.')[1].replace('.npy','')
        return float(f"{seconds}.{milliseconds}")
    except (IndexError, ValueError):
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
        file_path = os.path.join(folder_path, file)
        try:
            frame = np.load(file_path).astype(np.uint8)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
        if frame.shape[-1] != 3:
            continue
        angles = process_frame_split(frame)
        if angles:
            angles['time'] = timestamp
            data.append(angles)
    return pd.DataFrame(data)

def process_frame_split(frame):
   # Define keypoint indices (COCO format)
    NOSE = 0
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
    with contextlib.redirect_stdout(io.StringIO()):
        results = pose_model(frame, verbose=False)
    for result in results:
        keypoints = result.keypoints.data.cpu().numpy().flatten()
        def get_point(index):
            return (keypoints[index * 3], keypoints[index * 3 + 1])
        # For simplicity, we use the COCO indices (adjust as needed)
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

def resample_split(df, target_fs=10):
    if len(df) < 2 or 'time' not in df.columns:
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

def process_camera_folder(camera_folder, target_fs=10):
    """
    Process a single camera folder (e.g. camera_1) and return a DataFrame
    with pose/keypoint features.
    """
    df_pose = process_video_folder_split(camera_folder)
    if not df_pose.empty:
        df_resampled = resample_split(df_pose, target_fs=target_fs)
        return df_resampled
    return pd.DataFrame()

########################################
# Combined Processing: EMG + Camera Together
########################################

def combined_processing():
    # Base directory containing <uuid>/<action>/...
    base_dir = "/data1/dnicho26/EMG_DATASET/data/data"  
    # Output directory where combined data will be saved.
    output_dir = "/data1/dnicho26/EMG_DATASET/final-data-combined"
    os.makedirs(output_dir, exist_ok=True)

    # The combined index will hold one entry per (uuid, action)
    combined_index = []

    # Iterate over each UUID folder
    uuid_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    for uuid in tqdm(uuid_folders, desc="Processing UUIDs"):
        uuid_path = os.path.join(base_dir, uuid)
        # Each UUID folder may have one or more action folders
        action_folders = [d for d in os.listdir(uuid_path) if os.path.isdir(os.path.join(uuid_path, d))]
        for action in action_folders:
            action_path = os.path.join(uuid_path, action)
            # Define output directories for EMG and camera data
            out_emg_dir = os.path.join(output_dir, "emg", uuid, action)
            out_camera_dir = os.path.join(output_dir, "camera", uuid, action)
            os.makedirs(out_emg_dir, exist_ok=True)
            os.makedirs(out_camera_dir, exist_ok=True)

            # Process EMG CSV files (located directly in the action folder)
            emg_csv_files = glob(os.path.join(action_path, "*.csv"))
            emg_outputs = {"DS1": [], "DS2": [], "DS3": [], "DS4": []}
            for csv_file in emg_csv_files:
                process_csv_file(csv_file, input_base_dir=action_path, output_base_dir=out_emg_dir)
                # We assume that processed files are saved under out_emg_dir/<DS#>/<filename>
                ds1_file = os.path.join(out_emg_dir, "DS1", os.path.basename(csv_file))
                ds2_file = os.path.join(out_emg_dir, "DS2", os.path.basename(csv_file))
                ds3_file = os.path.join(out_emg_dir, "DS3", os.path.basename(csv_file))
                ds4_file = os.path.join(out_emg_dir, "DS4", os.path.basename(csv_file))
                if os.path.exists(ds1_file):
                    emg_outputs["DS1"].append(os.path.relpath(ds1_file, start=output_dir))
                if os.path.exists(ds2_file):
                    emg_outputs["DS2"].append(os.path.relpath(ds2_file, start=output_dir))
                if os.path.exists(ds3_file):
                    emg_outputs["DS3"].append(os.path.relpath(ds3_file, start=output_dir))
                if os.path.exists(ds4_file):
                    emg_outputs["DS4"].append(os.path.relpath(ds4_file, start=output_dir))

            # Process camera folders (those starting with "camera_") in the action folder
            camera_folders = [d for d in os.listdir(action_path) if d.lower().startswith("camera_") and os.path.isdir(os.path.join(action_path, d))]
            camera_outputs = []
            for cam in camera_folders:
                cam_folder = os.path.join(action_path, cam)
                df_cam = process_camera_folder(cam_folder, target_fs=10)
                if not df_cam.empty:
                    out_cam_file = os.path.join(out_camera_dir, f"{cam}_camera.csv")
                    df_cam.to_csv(out_cam_file, index=False)
                    camera_outputs.append(os.path.relpath(out_cam_file, start=output_dir))
                    print(f"Processed camera data for {cam_folder} saved to {out_cam_file}")
                else:
                    print(f"No valid camera data found in {cam_folder}")

            # Add a combined index entry if at least one modality was processed.
            if any(emg_outputs.values()) or camera_outputs:
                combined_index.append({
                    'uuid': uuid,
                    'action': action,
                    'emg_DS1': ";".join(emg_outputs["DS1"]),
                    'emg_DS2': ";".join(emg_outputs["DS2"]),
                    'emg_DS3': ";".join(emg_outputs["DS3"]),
                    'emg_DS4': ";".join(emg_outputs["DS4"]),
                    'camera': ";".join(camera_outputs)
                })

    # Save the combined index.
    combined_index_df = pd.DataFrame(combined_index)
    index_out = os.path.join(output_dir, "combined_index.csv")
    combined_index_df.to_csv(index_out, index=False)
    print(f"Combined index saved to {index_out}")

    # Build train/val/test splits based on UUID (the same split applies to all modalities)
    uuids = combined_index_df['uuid'].unique().tolist()
    random.shuffle(uuids)
    n = len(uuids)
    n_train = int(n * 0.7)
    n_val = int(n * 0.1)
    train_uuids = set(uuids[:n_train])
    val_uuids = set(uuids[n_train:n_train+n_val])
    def get_split(u):
        if u in train_uuids:
            return 'train'
        elif u in val_uuids:
            return 'val'
        else:
            return 'test'
    combined_index_df['split'] = combined_index_df['uuid'].apply(get_split)
    
    # Save split index files.
    for split in ['train', 'val', 'test']:
        split_df = combined_index_df[combined_index_df['split'] == split]
        split_out = os.path.join(output_dir, f"combined_index_{split}.csv")
        split_df.to_csv(split_out, index=False)
        print(f"Combined index for {split} saved to {split_out}")

if __name__ == "__main__":
    combined_processing()
