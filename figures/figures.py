import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

def segment_windows(df, window_size, overlap_fraction):
    step = window_size - int(overlap_fraction * window_size)
    segments = []
    for start in range(0, len(df) - window_size + 1, step):
        segments.append(df.iloc[start:start + window_size].reset_index(drop=True))
    return segments

def plot_windows(windows, fs_emg, fs_acc, skip_windows, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    windows = windows[skip_windows:]
    if not windows:
        print("❌ No windows left after skipping.")
        return

    window = windows[0]
    duration = window.shape[0] / fs_emg
    t_emg = np.arange(window.shape[0]) / fs_emg
    n_acc = int(round(duration * fs_acc))
    t_acc = np.linspace(0, duration, n_acc)

    modalities = {
        'emg':  [c for c in window.columns if 'emg'  in c.lower()],
        'acc':  [c for c in window.columns if 'acc'  in c.lower()],
        'gyro': [c for c in window.columns if 'gyro' in c.lower()],
    }

    for modality, cols in modalities.items():
        if len(cols) < 6:
            continue
        leg1, leg2 = cols[:3], cols[3:6]

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"{modality.upper()} – Window #{skip_windows+1}")

        if modality == 'emg':
            data1 = window[leg1].values; data2 = window[leg2].values
            t1 = t2 = t_emg
        else:
            # interpolate each channel onto t_acc
            data1 = np.stack([np.interp(t_acc, t_emg, window[c].values) for c in leg1], axis=1)
            data2 = np.stack([np.interp(t_acc, t_emg, window[c].values) for c in leg2], axis=1)
            t1 = t2 = t_acc

        for i in range(3):
            axs[0].plot(t1, data1[:, i], label=f"Sensor {i}")
            axs[1].plot(t2, data2[:, i], label=f"Sensor {i+3}")

        axs[0].set_title(f"{modality.upper()} Channels 0–2")
        axs[0].set_xlabel("Time (s)"); axs[0].set_ylabel("Value"); axs[0].legend()
        axs[1].set_title(f"{modality.upper()} Channels 3–5")
        axs[1].set_xlabel("Time (s)"); axs[1].set_ylabel("Value"); axs[1].legend()

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        out = os.path.join(output_dir, f"{modality}_window.png")
        fig.savefig(out); plt.close(fig)
        print(f"✅ Saved {modality.upper()} plot: {out}")
def main():
    base_dir = "/data1/dnicho26/EMG_DATASET/final-data"
    windows_outdir = (
        "/data1/dnicho26/Thesis/"
        "AI-Assisted-Gait-Restoration-for-Disabled-Individuals/"
        "figures/final-data/windows"
    )
    fs_emg = 10 #1259.259
    fs_acc = 10 #148.148
    skip_windows = 0

    # 1) Get all subject folders
    subj_folders = sorted(glob(os.path.join(base_dir, "*")))
    if not subj_folders:
        print("❌ No subject directories under", base_dir)
        return

    # 2) Pick the first subject
    subj_dir = subj_folders[0]
    print("ℹ️  Using subject folder:", subj_dir)

    # 3) Find action subfolders (exclude camera_*)
    action_dirs = [
        d for d in sorted(glob(os.path.join(subj_dir, "*")))
        if "camera_" not in os.path.basename(d).lower()
    ]
    if not action_dirs:
        print("❌ No action folders under", subj_dir)
        return

    # 4) Search each action for a CSV (recursive)
    csv_file = None
    for act in action_dirs:
        found = glob(os.path.join(act, "**", "*.csv"), recursive=True)
        if found:
            csv_file = found[0]
            print("ℹ️  Found CSV:", csv_file)
            break

    if not csv_file:
        print("❌ No CSV found in any action folder.")
        return

    # 5) Load and segment
    df = pd.read_csv(csv_file)
    window_size      = int(fs_emg * 3)  # 3 s @ EMG rate
    overlap_fraction = 0.0
    windows = segment_windows(df, window_size, overlap_fraction)
    print(f"🔍 {len(windows)} windows before skipping the first {skip_windows}")

    if not windows:
        print("❌ Not enough data for even one window.")
        return

    # 6) Plot EMG, ACC, GYRO aligned
    plot_windows(
        windows,
        fs_emg=fs_emg,
        fs_acc=fs_acc,
        skip_windows=skip_windows,
        output_dir=windows_outdir
    )

if __name__ == "__main__":
    main()
