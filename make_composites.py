#!/usr/bin/env python3
"""
make_composites.py

Generate final metrics from loss_summary logs and group composite figures
per trial and loss type, using configured model groups, saving under trials/figures.

Part 0: Final metrics summary (last epoch values), overall and per-loss
Part 1: Model composites:
    - loss curves
    - sample predictions
    - latest-epoch windows
    - validation and test result visualizations
    - copy sample prediction PNG
Part 2: Bar charts for RMSE, MAE, R², and correlation coefficient, and save metrics CSV.

Usage:
    python make_composites.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
from pathlib import Path

# --- Paths and setup ---
SCRIPT_DIR = Path("/data1/dnicho26/Thesis/AI-Assisted-Gait-Restoration-for-Disabled-Individuals")
EMG_ROOT   = SCRIPT_DIR / "trials" / "emg"
FIG_ROOT   = SCRIPT_DIR / "trials" / "figures"
FIG_ROOT.mkdir(parents=True, exist_ok=True)

# --- Configuration ---
TRIALS = [1, 10,15,20]
HEADER = ["epoch", "train_loss", "val_loss", "rmse", "mae", "r2_score", "corr_coef"]

def plot_grid(items, title, outpath, cols=3):
    if not items:
        return

    cols = min(len(items), cols)
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 2.5 * rows))
    axes = axes.flatten() if hasattr(axes, '__iter__') else [axes]

    for ax, (img, _) in zip(axes, items):
        image = imread(str(img))

        if "window" in img.name:
            crop_top = int(image.shape[0] * 0.02)
            image = image[crop_top:, :, :]
            suptitle = True
        elif "loss_curve" in str(outpath):
            suptitle = True
        else:
            suptitle = False

        ax.imshow(image)
        ax.axis('off')

    for ax in axes[len(items):]:
        ax.axis('off')

    plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.05, hspace=0.3, wspace=0.1)

    if suptitle:
        fig.suptitle(title, fontsize=10)

    fig.savefig(outpath, dpi=300)
    plt.close()

def plot_metric_barchart(trial, loss, metrics_dict):
    outdir = FIG_ROOT / f"trial_{trial}" / loss / "metrics"
    outdir.mkdir(parents=True, exist_ok=True)

    # Save metrics to CSV
    df_metrics = pd.DataFrame.from_dict(metrics_dict, orient='index')
    df_metrics.index.name = 'model'
    df_metrics.to_csv(outdir / "metrics_summary.csv")

    for metric in ['rmse', 'mae', 'r2_score', 'corr_coef']:
        fig, ax = plt.subplots(figsize=(max(6, len(metrics_dict) * 0.5), 4))
        names = list(metrics_dict.keys())
        vals = [metrics_dict[m][metric] for m in names]
        ax.bar(names, vals)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric.upper()} — Trial {trial}, Loss {loss}")
        ax.set_xticklabels(names, rotation=45, ha='right')
        plt.tight_layout()
        fig.savefig(outdir / f"{metric}_barchart.png", dpi=300)
        plt.close(fig)

# --- Part 1: Per-model composite plots and copying sample predictions ---
for trial in TRIALS:
    loss_types = {
        d.name
        for md in EMG_ROOT.iterdir() if (md / str(trial)).is_dir()
        for d in (md / str(trial)).iterdir() if d.is_dir()
    }

    for loss in sorted(loss_types):
        for model_dir in EMG_ROOT.iterdir():
            if not model_dir.is_dir():
                continue

            model = model_dir.name
            base = model_dir / str(trial) / loss
            if not base.is_dir():
                continue

            out_dir = FIG_ROOT / f"trial_{trial}" / loss / "models" / model
            out_dir.mkdir(parents=True, exist_ok=True)

            items_loss = []
            items_sample = []
            items_windows = []
            items_valres = []
            items_testres = []

            # 1) Loss curves
            loss_curve_files = (
                list(base.glob("*mse*.png")) +
                list(base.glob("*huber*.png")) +
                list(base.glob("loss_curve.png"))
            )
            for lc in loss_curve_files:
                items_loss.append((lc, f"{model.upper()}: {trial} ahead"))

            # 2) Sample predictions (copy and plot)
            sp_file = base / "figures" / f"{model}_sample_predictions.png"
            if sp_file.is_file():
                items_sample.append((sp_file, f"{model.upper()}: {trial} ahead"))
                try:
                    (out_dir / f"{model}_sample_predictions.png").write_bytes(sp_file.read_bytes())
                except Exception as e:
                    print(f"Failed to copy {sp_file}: {e}")

            # 3) Epoch windows (latest)
            figdir = base / 'figures' / model
            if figdir.is_dir():
                epochs = [d for d in figdir.iterdir() if d.is_dir() and 'epoch_' in d.name]
                if epochs:
                    latest = sorted(epochs, key=lambda d: int(d.name.split('_')[-1]))[-1]
                    for w in latest.glob("window_1_epoch_*.png"):
                        items_windows.append((w, f"{model.upper()}: {trial} ahead"))

            # 4) Validation results
            vdir = base / 'validation_results'
            if vdir.is_dir():
                for p in vdir.glob('*.png'):
                    items_valres.append((p, f"{model.upper()}: {trial} ahead"))

            # 5) Test results
            tdir = base / 'test_results'
            if tdir.is_dir():
                for p in tdir.glob('*.png'):
                    items_testres.append((p, f"{model.upper()}: {trial} ahead"))

            # Save plots
            if items_loss:
                plot_grid(items_loss, f"{model.upper()}: {trial} n ahead", out_dir / "loss_curves.png")
            if items_sample:
                plot_grid(items_sample, "", out_dir / "sample_predictions.png")
            if items_windows:
                plot_grid(items_windows, "", out_dir / "epoch_window_1.png")
            if items_valres:
                plot_grid(items_valres, "", out_dir / "validation_results.png")
            if items_testres:
                plot_grid(items_testres, "", out_dir / "test_results.png")

# --- Part 2: Metric bar charts and CSV saving ---
for trial in TRIALS:
    loss_types = {
        d.name
        for md in EMG_ROOT.iterdir() if (md / str(trial)).is_dir()
        for d in (md / str(trial)).iterdir() if d.is_dir()
    }

    for loss in sorted(loss_types):
        metrics_summary = {}
        for model_dir in EMG_ROOT.iterdir():
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            summary_path = model_dir / str(trial) / loss / "logs" / f"loss_summary_{model}.txt"
            if not summary_path.is_file():
                continue

            try:
                df = pd.read_csv(summary_path)
                last = df.iloc[-1]
                metrics_summary[model] = {
                    'rmse': last['rmse'],
                    'mae': last['mae'],
                    'r2_score': last['r2_score'],
                    'corr_coef': last['corr_coef'],
                }
            except Exception as e:
                print(f"Could not parse {summary_path}: {e}")

        if metrics_summary:
            plot_metric_barchart(trial, loss, metrics_summary)
