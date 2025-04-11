import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

def segment_windows(df, window_size, overlap_fraction):
    """
    Segments the DataFrame into windows of fixed length (in number of samples)
    with the specified fractional overlap.
    """
    step = window_size - int(overlap_fraction * window_size)
    segments = []
    for start in range(0, len(df) - window_size + 1, step):
        segment = df.iloc[start:start + window_size].reset_index(drop=True)
        segments.append(segment)
    return segments
def plot_windows(windows, num_windows=5, output_dir="final-data/figures/windows"):
    """
    Plots the first `num_windows` windows.
    Each window gets one PNG with 4 subplots:
      - EMG channels 0–2
      - ACC channels 0–2
      - GYRO channels 0–2
      - EMG channels 3–5
    """
    os.makedirs(output_dir, exist_ok=True)

    for i in range(min(num_windows, len(windows))):
        window = windows[i]

        # Identify sensor columns
        emg_cols = [col for col in window.columns if "emg" in col.lower()]
        acc_cols = [col for col in window.columns if "acc" in col.lower()]
        gyro_cols = [col for col in window.columns if "gyro" in col.lower()]

        # Select indices
        emg_0_2 = emg_cols[:3]
        emg_3_5 = emg_cols[3:6]
        acc_0_2 = acc_cols[:3]
        gyro_0_2 = gyro_cols[:3]

        # Create a single figure with 4 stacked subplots
        fig, axs = plt.subplots(4, 1, figsize=(12, 10))
        fig.suptitle(f"Window {i+1} - Sensor Overview")

        if emg_0_2:
            axs[0].plot(window[emg_0_2])
            axs[0].set_title("EMG Channels 0–2")
        else:
            axs[0].axis("off")

        if acc_0_2:
            axs[1].plot(window[acc_0_2])
            axs[1].set_title("ACC Channels 0–2")
        else:
            axs[1].axis("off")

        if gyro_0_2:
            axs[2].plot(window[gyro_0_2])
            axs[2].set_title("GYRO Channels 0–2")
        else:
            axs[2].axis("off")

        if emg_3_5:
            axs[3].plot(window[emg_3_5])
            axs[3].set_title("EMG Channels 3–5")
        else:
            axs[3].axis("off")

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = os.path.join(output_dir, f"window_{i+1}.png")
        fig.savefig(out_path)
        plt.close(fig)
        print(f"✅ Saved combined window plot: {out_path}")

def plot_histograms(csv_files, output_dir="final-data/figures/histograms"):
    """
    Aggregates all CSVs in the dataset and plots histograms of sensor distributions.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_data = []
    for f in tqdm(csv_files, desc="Loading CSVs for histograms"):
        try:
            df = pd.read_csv(f)
            all_data.append(df)
        except Exception as e:
            print(f"Failed to load {f}: {e}")

    if not all_data:
        print("No valid data loaded for histogram plotting.")
        return

    df_full = pd.concat(all_data, axis=0, ignore_index=True)

    modalities = {'emg': [], 'acc': [], 'gyro': []}
    for col in df_full.columns:
        col_lower = col.lower()
        if "emg" in col_lower:
            modalities['emg'].append(col)
        elif "acc" in col_lower:
            modalities['acc'].append(col)
        elif "gyro" in col_lower:
            modalities['gyro'].append(col)

    for modality, cols in modalities.items():
        if cols:
            data = df_full[cols].values.flatten()
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(data, bins=50)
            ax.set_title(f"Histogram of {modality.upper()} values (entire dataset)")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            fig.tight_layout()
            filepath = os.path.join(output_dir, f"histogram_{modality}.png")
            fig.savefig(filepath)
            plt.close(fig)
            print(f"Saved {modality.upper()} histogram to {filepath}")

    for axis in ['acc_x', 'acc_y', 'acc_z']:
        cols = [col for col in df_full.columns if axis in col.lower()]
        if cols:
            data = df_full[cols].values.flatten()
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(data, bins=50)
            ax.set_title(f"Histogram of {axis.upper()} values (entire dataset)")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            fig.tight_layout()
            filepath = os.path.join(output_dir, f"histogram_{axis}.png")
            fig.savefig(filepath)
            plt.close(fig)
            print(f"Saved {axis.upper()} histogram to {filepath}")

def main():
    base_dir = "/data1/dnicho26/EMG_DATASET/final-data/"
    figures_dir = "/data1/dnicho26/Thesis/AI-Assisted-Gait-Restoration-for-Disabled-Individuals/figures/dataset"
    windows_outdir = os.path.join(figures_dir, "windows")
    histograms_outdir = os.path.join(figures_dir, "histograms")

    csv_files = glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)
    if not csv_files:
        print("No CSV files found in processed dataset directory.")
        return

    # Load first sample CSV for window plotting
    sample_df = pd.read_csv(csv_files[0])
    window_size = 30  # 3 seconds @ 10 Hz
    overlap_fraction = 0.5
    windows = segment_windows(sample_df, window_size, overlap_fraction)

    if windows:
        print(f"Segmented into {len(windows)} windows.")
        plot_windows(windows, num_windows=10, output_dir=windows_outdir)
    else:
        print("Not enough data to segment into windows.")

    # Plot histograms over the entire dataset
    plot_histograms(csv_files, output_dir=histograms_outdir)

if __name__ == "__main__":
    main()
