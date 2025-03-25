import os
import re
import pandas as pd
import numpy as np
from scipy.signal import filtfilt, butter, resample_poly
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from fractions import Fraction
from sklearn.preprocessing import RobustScaler
import joblib

def remove_imp_cols_and_unnamed(df):
    df = df.loc[:, ~df.columns.str.contains('IMP', case=False)]
    df = df.loc[:, ~df.columns.str.contains('Unnamed', case=False)]
    return df

def lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def highpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

def replace_outliers_zscore(x, threshold=3):
    mean_val = np.nanmean(x)
    std_val = np.nanstd(x)
    if std_val == 0 or np.isnan(std_val):
        return x
    z = np.abs((x - mean_val) / std_val)
    median_val = np.nanmedian(x)
    x[z >= threshold] = median_val
    return x

def replace_outliers_mad(x, threshold=3.5):
    median_val = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - median_val))
    if mad == 0:
        return x
    modified_z = 0.6745 * (x - median_val) / mad
    x[np.abs(modified_z) >= threshold] = median_val
    return x

def replace_outliers_iqr(x, factor=1.5):
    q1 = np.nanpercentile(x, 25)
    q3 = np.nanpercentile(x, 75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    median_val = np.nanmedian(x)
    x[(x < lower_bound) | (x > upper_bound)] = median_val
    return x

def causal_moving_average_1d(signal, window=5):
    out = np.zeros_like(signal)
    cum_sum = 0
    for i in range(len(signal)):
        cum_sum += signal[i]
        if i >= window:
            cum_sum -= signal[i - window]
            out[i] = cum_sum / window
        else:
            out[i] = cum_sum / (i + 1)
    return out

def resample_emg_keep_inertial(df):
    cols = list(df.columns)
    emg_cols = [col for col in cols if "emg" in col.lower()]
    inertial_cols = [col for col in cols if ("acc" in col.lower() or "gyro" in col.lower())]
    if not emg_cols or not inertial_cols:
        return df

    df_inertial_valid = df[inertial_cols].dropna(how='any')
    inertial_len = len(df_inertial_valid)
    if inertial_len == 0:
        return df

    df_emg_valid = df[emg_cols].dropna(how='all')
    emg_len = len(df_emg_valid)
    if emg_len == 0:
        return df

    ratio = inertial_len / emg_len
    frac = Fraction(ratio).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator

    emg_array = df[emg_cols].values
    emg_resampled = resample_poly(emg_array, up, down, axis=0)
    emg_resampled = emg_resampled[:inertial_len, :]

    df_emg_resampled = pd.DataFrame(emg_resampled, columns=emg_cols).reset_index(drop=True)
    df_inertial_resampled = df[inertial_cols].iloc[:inertial_len].reset_index(drop=True)
    new_df = pd.concat([df_inertial_resampled, df_emg_resampled], axis=1)
    new_df = new_df[[col for col in cols if col in new_df.columns]]
    new_df.reset_index(drop=True, inplace=True)
    return new_df

def manual_min_max_scale(vals, feature_range=(-1, 1)):
    min_val = np.min(vals)
    max_val = np.max(vals)
    # Avoid division by zero if all values are equal.
    if max_val == min_val:
        return np.zeros_like(vals)
    a, b = feature_range
    return (vals - min_val) / (max_val - min_val) * (b - a) + a

def process_file(file_path, data_dir, fs_target=148.148, emg_scale=100.0,
                 emg_cutoff=20, inertial_cutoff=5, gyro_highpass_cutoff=0.5,
                 outlier_method="mad", causal_window=5,
                 debug_std=False,
                 flag_scale=False, flag_outlier=True, flag_lowpass=False,
                 flag_moving_average=False, flag_highpass=True,
                 flag_normalize=True):  # New flag added here.
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return None

    file_name = os.path.basename(file_path)
    file_uuid = file_name.split('.')[0]
    parts = [p for p in file_path.split(os.sep) if p]
    action = parts[-2] if len(parts) >= 2 else "unknown"


    df = pd.read_csv(file_path)
    df = remove_imp_cols_and_unnamed(df)
    if 'time' in df.columns:
        df = df.drop(columns=['time'])
    df = resample_emg_keep_inertial(df)
    nan_columns = df.columns[df.isna().all()].tolist()
    if nan_columns:
        print(f"Warning: Entirely NaN columns in {file_path}: {nan_columns}")
        df = df.drop(columns=nan_columns)
    if len(df.columns) == 0:
        print(f"Error: No valid columns remain in {file_path} after dropping NaNs")
        return None

    df.attrs['file_uuid'] = file_uuid
    df.attrs['action'] = action

    processed_dict = {}
    # Loop through each column and decide how to process.
    for col in df.columns:
        # Ensure we work with a numeric array.
        vals = df[col].values.astype(np.float32)
        vals = np.nan_to_num(vals, nan=0.0)


        # Remove outliers based on the specified method.
        if flag_outlier:
            if outlier_method == "zscore":
                vals = replace_outliers_zscore(vals)
            elif outlier_method == "mad":
                vals = replace_outliers_mad(vals)
            elif outlier_method == "iqr":
                vals = replace_outliers_iqr(vals)
                    
        from sklearn.preprocessing import MinMaxScaler
        # If our simple flag is set, just remove outliers and normalize.
        if flag_normalize:
            # Normalize manually to the range [-1, 1]
            vals = manual_min_max_scale(vals, feature_range=(-1, 1))
            processed_dict[col] = vals

        # Otherwise, process columns as before.
        name_lower = col.lower()
        if "emg" in name_lower:
            if flag_scale:
                # Adaptive scaling based on maximum absolute values.
                max_values = []
                for emg_col in [c for c in df.columns if "emg" in c.lower()]:
                    temp_vals = df[emg_col].values.astype(np.float32)
                    temp_vals = np.nan_to_num(temp_vals, nan=0.0)
                    max_values.append(np.max(np.abs(temp_vals)))
                if max(max_values) > 50:
                    if max(max_values) > 10000:
                        adaptive_scale = 1/10000
                    elif max(max_values) > 1000:
                        adaptive_scale = 1/1000
                    elif max(max_values) > 100:
                        adaptive_scale = 1/100
                    else:
                        adaptive_scale = 1.0
                else:
                    adaptive_scale = emg_scale
                vals = vals * adaptive_scale
            vals = np.abs(vals)
            if flag_outlier:
                if outlier_method == "zscore":
                    vals = replace_outliers_zscore(vals)
                elif outlier_method == "mad":
                    vals = replace_outliers_mad(vals)
                elif outlier_method == "iqr":
                    vals = replace_outliers_iqr(vals)
            if flag_lowpass:
                vals = lowpass_filter(vals, cutoff=emg_cutoff, fs=fs_target, order=4)
            if flag_moving_average:
                vals = causal_moving_average_1d(vals, window=causal_window)
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            if debug_std:
                print(f"{file_uuid} - {col}: mean={mean_val:.2f}, std={std_val:.2f}")
            if np.max(vals) > 10:
                print(f"{file_uuid} - {col}: max {np.max(vals):.2f} exceeds threshold. Dropping CSV: {file_path}")
                return None
            if std_val > 10:
                print(f"{file_uuid} - {col}: std {std_val:.2f} exceeds threshold. Dropping CSV: {file_path}")
                return None

        elif "acc" in name_lower:
            vals = np.abs(vals)
            if flag_outlier:
                if outlier_method == "zscore":
                    vals = replace_outliers_zscore(vals)
                elif outlier_method == "mad":
                    vals = replace_outliers_mad(vals)
                elif outlier_method == "iqr":
                    vals = replace_outliers_iqr(vals)
            if flag_lowpass:
                vals = lowpass_filter(vals, cutoff=inertial_cutoff, fs=fs_target, order=4)
            if flag_moving_average:
                vals = causal_moving_average_1d(vals, window=causal_window)

        elif "gyro" in name_lower:
            vals = np.abs(vals)
            if flag_outlier:
                if outlier_method == "zscore":
                    vals = replace_outliers_zscore(vals)
                elif outlier_method == "mad":
                    vals = replace_outliers_mad(vals)
                elif outlier_method == "iqr":
                    vals = replace_outliers_iqr(vals)
            if flag_highpass:
                vals = highpass_filter(vals, cutoff=gyro_highpass_cutoff, fs=fs_target, order=4)
            if flag_lowpass:
                vals = lowpass_filter(vals, cutoff=inertial_cutoff, fs=fs_target, order=4)
        
        if debug_std:
            print(f"{file_uuid} - {col} after processing: mean={np.mean(vals):.2f}, std={np.std(vals):.2f}")
        processed_dict[col] = vals

    df_processed = pd.DataFrame(processed_dict)
    os.makedirs(data_dir, exist_ok=True)
    processed_file_path = os.path.join(data_dir, file_name)
    df_processed.to_csv(processed_file_path, index=False)
    return (df_processed, processed_file_path)



def plot_modality_single_file(df, modality, dataset_type, figure_dir):
    modality = modality.lower()
    modality_cols = [col for col in df.columns if modality in col.lower()]
    if not modality_cols:
        print(f"No columns found for modality: {modality}")
        return

    sensor_pattern = re.compile(r'sensor\s*(\d+)', re.IGNORECASE)
    sensor_groups = {}
    for col in modality_cols:
        match = sensor_pattern.search(col)
        sensor_num = match.group(1) if match else "unknown"
        sensor_groups.setdefault(sensor_num, []).append(col)

    sensor_keys = sorted(sensor_groups.keys(), key=lambda x: int(x) if x.isdigit() else x)
    n_sensors = len(sensor_keys)
    fig, axs = plt.subplots(n_sensors, 1, figsize=(12, 3 * n_sensors), sharex=True)
    if n_sensors == 1:
        axs = [axs]
    for i, sensor in enumerate(sensor_keys):
        ax = axs[i]
        for col in sensor_groups[sensor]:
            ax.plot(df[col], label=col)
        ax.set_title(f"Sensor {sensor}")
        ax.legend()
    action = df.attrs.get('action', 'unknown')
    file_uuid = df.attrs.get('file_uuid', 'unknown')
    fig.suptitle(f"{dataset_type.upper()} {modality.upper()} - Action: {action}, UUID: {file_uuid}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    file_name = f"{dataset_type.lower()}_{modality}_{action}_{file_uuid}.png"
    full_path = os.path.join(figure_dir, "single_sensor", file_name)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    plt.savefig(full_path)
    plt.close()
    print(f"Saved single file modality plot: {full_path}")

def plot_modality(df, modality, save_filename, title_prefix=""):
    modality = modality.lower()
    modality_cols = [col for col in df.columns if modality in col.lower()]
    if not modality_cols:
        print(f"No columns found for modality: {modality}")
        return
    sensor_pattern = re.compile(r'sensor\s*(\d+)', re.IGNORECASE)
    sensor_groups = {}
    for col in modality_cols:
        match = sensor_pattern.search(col)
        key = match.group(1) if match else "unknown"
        sensor_groups.setdefault(key, []).append(col)
    n_sensors = len(sensor_groups)
    n_subplots = n_sensors + 1
    fig, axs = plt.subplots(n_subplots, 1, figsize=(12, 4 * n_subplots), sharex=True)
    if n_subplots == 1:
        axs = [axs]
    sensor_order = sorted(sensor_groups.keys(), key=lambda x: int(x) if x.isdigit() else x)
    for idx, sensor_num in enumerate(sensor_order):
        ax = axs[idx]
        for col in sensor_groups[sensor_num]:
            ax.plot(df[col], label=col)
        ax.set_title(f"{title_prefix} {modality.upper()} Sensor {sensor_num}")
        ax.set_ylabel("Value")
        ax.legend()
    ax_overlay = axs[-1]
    for sensor_num in sensor_order:
        sensor_data = df[sensor_groups[sensor_num]].mean(axis=1)
        ax_overlay.plot(sensor_data, label=f"Sensor {sensor_num}")
    ax_overlay.set_title(f"{title_prefix} {modality.upper()} Sensors Overlay")
    ax_overlay.set_xlabel("Sample")
    ax_overlay.set_ylabel("Mean Value")
    ax_overlay.legend()
    plt.tight_layout()
    plt.savefig(save_filename)
    plt.close()
    print(f"Saved plot: {save_filename}")

def plot_emg_gt_and_pred_windows(df, lag=60, n_ahead=5, save_filename="gt_vs_pred_windows.png"):
    emg_cols = [col for col in df.columns if "emg" in col.lower()]
    if len(emg_cols) < 6:
        print("Not enough EMG columns for GT vs Pred plotting.")
        return
    pred_cols = emg_cols[0:3]
    gt_cols = emg_cols[3:6]
    total_window = lag + n_ahead
    plt.figure(figsize=(12, 10))
    for i in range(5):
        start = i * total_window
        end = start + total_window
        if end > len(df):
            break
        window_time = np.arange(start, end)
        plt.subplot(5, 1, i+1)
        for col in gt_cols:
            plt.plot(window_time, df[col].values[start:end], label=f"GT {col}")
        for col in pred_cols:
            plt.plot(window_time, df[col].values[start:end], linestyle="--", label=f"Pred {col}")
        plt.title(f"Window {i+1} (Samples {start} to {end-1})")
        plt.legend()
    plt.tight_layout()
    plt.savefig(save_filename)
    plt.close()
    print(f"Saved GT vs Pred window plot: {save_filename}")

def fit_and_save_scalers(train_dfs, scaler_dir):
    os.makedirs(scaler_dir, exist_ok=True)
    scaler_dict = {}
    modalities = ["emg", "acc", "gyro"]
    for mod in modalities:
        intersect_cols = None
        for df in train_dfs:
            cols = [col for col in df.columns if mod in col.lower()]
            if intersect_cols is None:
                intersect_cols = set(cols)
            else:
                intersect_cols = intersect_cols.intersection(set(cols))
        if intersect_cols:
            col_list = sorted(list(intersect_cols))
            mod_data_list = []
            for df in train_dfs:
                mod_data_list.append(df[col_list])
            combined = pd.concat(mod_data_list, axis=0)
            scaler = RobustScaler().fit(combined.values)
            scaler_dict[mod] = (scaler, col_list)
            joblib.dump((scaler, col_list), os.path.join(scaler_dir, f"{mod}_scaler.pkl"))
            print(f"Saved {mod} scaler with {len(col_list)} features.")
    return scaler_dict

def load_scalers(scaler_dir):
    scaler_dict = {}
    for mod in ["emg", "acc", "gyro"]:
        scaler_path = os.path.join(scaler_dir, f"{mod}_scaler.pkl")
        if os.path.exists(scaler_path):
            scaler, col_list = joblib.load(scaler_path)
            scaler_dict[mod] = (scaler, col_list)
            print(f"Loaded {mod} scaler with {len(col_list)} features.")
    return scaler_dict

def apply_scalers(dfs, scaler_dict):
    for i, df in enumerate(dfs):
        for mod, (scaler, col_list) in scaler_dict.items():
            file_cols = [col for col in df.columns if mod in col.lower()]
            if set(file_cols) >= set(col_list):
                data = df[col_list].values
                df.loc[:, col_list] = scaler.transform(data)
            else:
                print(f"File {i}: Modality '{mod}' missing expected columns; skipping scaling for this modality.")
        dfs[i] = df
    return dfs

def make_window_plots(df, dataset_type, figure_dir, lag=60, n_ahead=5):
    windows_dir = os.path.join(figure_dir, "windows")
    os.makedirs(windows_dir, exist_ok=True)
    filename = os.path.join(windows_dir, f"{dataset_type}_gt_vs_pred_windows.png")
    plot_emg_gt_and_pred_windows(df, lag=lag, n_ahead=n_ahead, save_filename=filename)

def make_dataset_plot(dfs, dataset_type, figure_dir):
    dataset_dir = os.path.join(figure_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    entire_df = pd.concat(dfs, axis=0, ignore_index=True)
    for mod in ["emg", "acc", "gyro"]:
        filename = os.path.join(dataset_dir, f"{dataset_type}_{mod}.png")
        plot_modality(entire_df, mod, filename, title_prefix=f"{dataset_type} Entire Dataset")

def process_dataset(index_file, dataset_type, data_dir, debug_std=False):
    dfs = []
    processed_paths = []
    dropped = 0
    if not os.path.isfile(index_file):
        print(f"Index file not found: {index_file}")
        return dfs, processed_paths, dropped
    df_idx = pd.read_csv(index_file)
    if "emg_file" not in df_idx.columns:
        print(f"Index file {index_file} missing 'emg_file' column.")
        return dfs, processed_paths, dropped
    file_paths = df_idx["emg_file"].tolist()
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_file = {executor.submit(process_file, file_path, data_dir, debug_std=debug_std): file_path for file_path in file_paths}
        for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc=f"Processing {dataset_type}"):
            result = future.result()
            if result is not None:
                df_processed, processed_file_path = result
                dfs.append(df_processed)
                processed_paths.append(processed_file_path)
            else:
                dropped += 1
    print(f"{dataset_type}: Processed {len(dfs)} files; Dropped {dropped} files.")
    return dfs, processed_paths, dropped


def update_index_file(original_index, processed_paths, output_index_path):
    import os
    # Read the original index file
    df_idx = pd.read_csv(original_index)
    
    # Create a new 'action' column from the original file paths
    df_idx["action"] = df_idx["emg_file"].apply(
        lambda x: os.path.basename(os.path.dirname(x)) if pd.notna(x) else np.nan
    )
    
    # Build a mapping from the base filename (from the original file) to the new processed file path.
    processed_map = {os.path.basename(path): path for path in processed_paths}
    
    # Update the 'emg_file' column with the new processed file path
    df_idx["emg_file"] = df_idx["emg_file"].apply(
        lambda orig: processed_map.get(os.path.basename(orig), np.nan)
    )
    
    # Drop rows that were not successfully processed.
    df_idx = df_idx.dropna(subset=["emg_file"]).reset_index(drop=True)
    
    df_idx.to_csv(output_index_path, index=False)
    print(f"Updated index file saved to {output_index_path}")



def main():
    base_dir = "/data1/dnicho26/EMG_DATASET/data/processed-server"
    append_action = "" 

    train_index = os.path.join(base_dir, f"index_train{append_action}.csv")
    val_index   = os.path.join(base_dir, f"index_val{append_action}.csv")
    test_index  = os.path.join(base_dir, f"index_test{append_action}.csv")

    # Directories for processed CSV files and updated index files.
    data_dir = "/data1/dnicho26/EMG_DATASET/data/final-proc-server/data"
    index_files_dir = "/data1/dnicho26/EMG_DATASET/data/final-proc-server/index_files"
    figure_dir = "/data1/dnicho26/Thesis/AI-Assisted-Gait-Restoration-for-Disabled-Individuals/figures/processing"
    scaler_dir = "/data1/dnicho26/Thesis/AI-Assisted-Gait-Restoration-for-Disabled-Individuals/models/scaler"
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(index_files_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    os.makedirs(scaler_dir, exist_ok=True)
    debug_std = False
    apply_scaling = False
    plotting=True

    print("Processing TRAIN dataset...")
    train_dfs, train_paths, train_dropped = process_dataset(train_index, "TRAIN", data_dir, debug_std)
    if train_dfs:
        if apply_scaling:
            scaler_dict = fit_and_save_scalers(train_dfs, scaler_dir)
            train_dfs = apply_scalers(train_dfs, scaler_dict)
        if plotting:
            for mod in ["emg", "acc", "gyro"]:
                plot_modality_single_file(train_dfs[0], mod, "TRAIN", figure_dir)

            make_window_plots(train_dfs[0], "TRAIN", figure_dir)
            make_dataset_plot(train_dfs, "TRAIN", figure_dir)

        # Update the train index with processed file paths
        updated_train_index = os.path.join(index_files_dir, f"index_train{append_action}.csv")
        update_index_file(train_index, train_paths, updated_train_index)
    else:
        print("No TRAIN files processed.")
    
    print("Processing VAL dataset...")
    val_dfs, val_paths, val_dropped = process_dataset(val_index, "VAL", data_dir, debug_std)
    if val_dfs:
        if apply_scaling:
            loaded_scalers = load_scalers(scaler_dir)
            val_dfs = apply_scalers(val_dfs, loaded_scalers)
        if plotting:
            for mod in ["emg", "acc", "gyro"]:
                plot_modality_single_file(val_dfs[0], mod, "VAL", figure_dir)
            make_window_plots(val_dfs[0], "VAL", figure_dir)
            make_dataset_plot(val_dfs, "VAL", figure_dir)

        updated_val_index = os.path.join(index_files_dir, f"index_val{append_action}.csv")
        update_index_file(val_index, val_paths, updated_val_index)
    else:
        print("No VAL files processed.")

    print("Processing TEST dataset...")
    test_dfs, test_paths, test_dropped = process_dataset(test_index, "TEST", data_dir, debug_std)
    if test_dfs:
        if apply_scaling:
            loaded_scalers = load_scalers(scaler_dir)
            test_dfs = apply_scalers(test_dfs, loaded_scalers)
        if plotting:
            for mod in ["emg", "acc", "gyro"]:
                plot_modality_single_file(test_dfs[0], mod, "TEST", figure_dir)
            make_window_plots(test_dfs[0], "TEST", figure_dir)
            make_dataset_plot(test_dfs, "TEST", figure_dir)
        updated_test_index = os.path.join(index_files_dir, f"index_test{append_action}.csv")
        update_index_file(test_index, test_paths, updated_test_index)
    else:
        print("No TEST files processed.")

    # Optionally, print out the lists of processed CSV paths
    print("Processed CSV paths:")
    print("TRAIN:", train_paths)
    print("VAL:", val_paths)
    print("TEST:", test_paths)

if __name__ == "__main__":
    main()
