#!/usr/bin/env python3
"""
benchmark_inference.py

Recursively finds all trained model checkpoints from experiment directories (trials/, trials-acc/, trials-not-norm/),
loads each model, runs a single dummy inference, and reports the inference time in milliseconds.

Assumes model architectures and input shapes are as in training.py/training-acc.py/training-non-normalized.py.
"""
import os
import re
import time
import torch
import numpy as np
from glob import glob
import sys
import csv

# Limit inference to one CPU core
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all model classes
from models.models import (
    LSTMModel, RNNModel, GRUModel, TCNModel, TemporalTransformer,
    TimeSeriesTransformer, Informer, NBeats, DBN,
    PatchTST, CrossFormer, DLinear, HybridLSTMTransformer
)

# Map model names to classes (update as needed)
MODEL_CLASSES = {
    "lstm": LSTMModel,
    "rnn": RNNModel,
    "gru": GRUModel,
    "tcn": TCNModel,
    "temporal_transformer": TemporalTransformer,
    "timeseries_transformer": TimeSeriesTransformer,
    "informer": Informer,
    "nbeats": NBeats,
    "dbn": DBN,
    "patchtst": PatchTST,
    "crossformer": CrossFormer,
    "dlinear": DLinear,
    "hybridlstmtransformer": HybridLSTMTransformer
}

# Default input shapes for each sensor mode (update as needed)
INPUT_SIZES = {"all": 21, "emg": 3, "acc": 9, "gyro": 9}
LAG = 30
N_AHEAD = 10
BATCH_SIZE = 1

# List all base trial directories
TRIAL_DIRS = [
    "../trials/", "../trials-acc/", "../trials-not-norm/", "../trials-features/"
]
TRIAL_DIRS = [os.path.abspath(os.path.join(os.path.dirname(__file__), d)) for d in TRIAL_DIRS if os.path.exists(os.path.join(os.path.dirname(__file__), d))]

# Utility to find best/last checkpoint in each experiment directory
def find_best_checkpoints(base_dirs):
    checkpoints = []
    for base in base_dirs:
        for root, dirs, files in os.walk(base):
            # Only consider directories with checkpoints
            ckpts = [f for f in files if f.endswith(".pt") or f.endswith(".pth")]
            if not ckpts:
                continue
            # Prefer best_model if present
            best = [f for f in ckpts if 'best_model' in f]
            if best:
                checkpoints.append(os.path.join(root, best[0]))
            else:
                # Otherwise, pick the checkpoint with the highest epoch number
                epoch_ckpts = []
                for f in ckpts:
                    m = re.search(r'epoch_(\d+)', f)
                    if m:
                        epoch_ckpts.append((int(m.group(1)), f))
                if epoch_ckpts:
                    epoch_ckpts.sort(reverse=True)
                    checkpoints.append(os.path.join(root, epoch_ckpts[0][1]))
                else:
                    # fallback: just pick the last checkpoint alphabetically
                    checkpoints.append(os.path.join(root, sorted(ckpts)[-1]))
    return checkpoints

def infer_model_class_from_path(path):
    """Infer model name from the path and return the class."""
    for name in MODEL_CLASSES.keys():
        if name in path.lower():
            return name, MODEL_CLASSES[name]
    return None, None

def infer_sensor_mode_from_path(path):
    for mode in INPUT_SIZES.keys():
        if f"/{mode}/" in path or f"_{mode}_" in path:
            return mode
    # fallback
    return "emg"

def get_n_ahead_from_path(path):
    m = re.search(r"/(\d{1,3})/", path)
    if m:
        return int(m.group(1))
    return N_AHEAD

def benchmark_inference():
    checkpoints = find_best_checkpoints(TRIAL_DIRS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for ckpt_path in checkpoints:
        model_name, model_cls = infer_model_class_from_path(ckpt_path)
        if model_cls is None:
            continue
        sensor_mode = infer_sensor_mode_from_path(ckpt_path)
        input_size = INPUT_SIZES[sensor_mode]
        n_ahead = get_n_ahead_from_path(ckpt_path)
        output_size = input_size if sensor_mode != "all" else 3

        model = None
        try:
            if model_name == "patchtst":
                model = model_cls(
                    input_channels=input_size,
                    seq_len=LAG,
                    forecast_horizon=n_ahead,
                    output_size=output_size
                ).to(device)
            elif model_name == "temporal_transformer":
                model = model_cls(
                    input_size=input_size,
                    seq_len=LAG,
                    d_model=64,
                    nhead=8,
                    num_layers=3,
                    dim_feedforward=256,
                    dropout=0.1,
                    forecast_horizon=n_ahead,
                    output_size=output_size
                ).to(device)
            elif model_name == "dbn":
                model = model_cls(
                    input_size=input_size,
                    hidden_size=128,
                    num_classes=output_size,
                    n_layers=3
                ).to(device)
            elif model_name == "tcn":
                model = model_cls(
                    input_channels=input_size,
                    num_channels=[32, 32],
                    kernel_size=3,
                    dropout=0.2,
                    num_classes=output_size,
                    n_ahead=n_ahead
                ).to(device)
            elif model_name in ["lstm", "gru", "rnn"]:
                model = model_cls(
                    input_size=input_size,
                    hidden_size=256,
                    num_layers=5,
                    num_classes=output_size,
                    n_ahead=n_ahead
                ).to(device)
            elif model_name == "dlinear":
                model = model_cls(
                    seq_len=LAG,
                    forecast_horizon=n_ahead,
                    num_channels=input_size,
                    individual=False,
                    moving_avg_kernel=25
                ).to(device)
            elif model_name == "crossformer":
                model = model_cls(
                    input_channels=input_size,
                    seq_len=LAG,
                    d_model=64,
                    nhead=8,
                    num_layers=3,
                    dim_feedforward=256,
                    dropout=0.1,
                    forecast_horizon=n_ahead,
                    output_size=output_size
                ).to(device)
            elif model_name == "hybridlstmtransformer":
                model = model_cls(
                    input_size=input_size,
                    lstm_hidden_size=256,
                    lstm_layers=2,
                    d_model=64,
                    nhead=8,
                    transformer_layers=2,
                    forecast_horizon=n_ahead,
                    output_size=output_size,
                    dropout=0.1
                ).to(device)
            elif model_name == "nbeats":
                model = model_cls(
                    input_size=input_size * LAG,
                    num_stacks=2,
                    num_blocks_per_stack=3,
                    num_layers=4,
                    hidden_size=256,
                    output_size=output_size,
                    n_ahead=n_ahead,
                    stack_types=("trend", "seasonality"),
                    share_weights_in_stack=False,
                    trend_degree=2,
                    seasonality_num_harmonics=8
                ).to(device)
            elif model_name == "timeseries_transformer":
                model = model_cls(
                    input_size=input_size,
                    seq_length=LAG,
                    num_layers=3,
                    d_model=64,
                    nhead=8,
                    dim_feedforward=256,
                    dropout=0.1,
                    forecast_horizon=n_ahead,
                    output_size=output_size
                ).to(device)
            elif model_name == "informer":
                model = model_cls(
                    enc_in=input_size,
                    dec_in=input_size,
                    c_out=output_size,
                    seq_len=LAG,
                    label_len=15,
                    out_len=n_ahead,
                    d_model=64,
                    n_heads=8,
                    e_layers=2,
                    d_layers=1,
                    d_ff=256,
                    dropout=0.1
                ).to(device)
            else:
                # fallback: try input_size, hidden_size, num_layers, num_classes, n_ahead
                model = model_cls(
                    input_size=input_size,
                    hidden_size=256,
                    num_layers=5,
                    num_classes=output_size,
                    n_ahead=n_ahead
                ).to(device)
        except Exception as e:
            print(f"[ERROR] Could not instantiate model for {ckpt_path}: {e}")
            continue
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("module.", "")
                new_state_dict[new_key] = v
            model.load_state_dict(new_state_dict, strict=False)
            model.eval()
        except Exception as e:
            print(f"[ERROR] Could not load checkpoint {ckpt_path}: {e}")
            continue
        dummy_input = torch.randn(BATCH_SIZE, LAG, input_size).to(device)
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy_input)
        torch.cuda.synchronize() if device == "cuda" else None
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        torch.cuda.synchronize() if device == "cuda" else None
        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000
        results.append({
            "model": model_name,
            "sensor_mode": sensor_mode,
            "n_ahead": n_ahead,
            "inference_ms": elapsed_ms,
            "ckpt": ckpt_path
        })
        print(f"Model: {model_name:25s} | Sensor: {sensor_mode:5s} | n_ahead: {n_ahead:2d} | Time: {elapsed_ms:8.3f} ms | Path: {os.path.basename(ckpt_path)}")
    # Save to CSV
    csv_path = os.path.join(os.path.dirname(__file__), "inference_benchmarks.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["model", "sensor_mode", "n_ahead", "inference_ms", "ckpt"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\nResults saved to {csv_path}")
    print("\n--- Inference Time Summary (ms) ---")
    for r in results:
        print(f"{r['model']:25s} | {r['sensor_mode']:5s} | n_ahead: {r['n_ahead']:2d} | {r['inference_ms']:8.3f} ms | {os.path.basename(r['ckpt'])}")

def simulate_realtime_inference_on_csv(csv_path=None):
    """
    Loads all best model checkpoints, loads a single CSV, and simulates real-time inference
    by feeding the CSV window-by-window to each model, printing predictions and timing.
    If csv_path is None, uses the same CSV as baseline.py.
    """
    import pandas as pd
    from utils.datasets import EMG_dataset
    import torch
    import time
    import os
    import numpy as np
    
    # Default CSV path as used in baseline.py
    DEFAULT_CSV = "/data1/dnicho26/EMG_DATASET/final-data/2/treadmill/1740952466.7207062.csv"
    if csv_path is None:
        csv_path = DEFAULT_CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    checkpoints = find_best_checkpoints(TRIAL_DIRS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latency_records = []
    for ckpt_path in checkpoints:
        model_name, model_cls = infer_model_class_from_path(ckpt_path)
        if model_cls is None:
            print(f"[SKIP] Could not infer model class for {ckpt_path}")
            continue
        sensor_mode = infer_sensor_mode_from_path(ckpt_path)
        input_size = INPUT_SIZES[sensor_mode]
        n_ahead = get_n_ahead_from_path(ckpt_path)
        output_size = input_size if sensor_mode != "all" else 3
        dataset = EMG_dataset(
            processed_index_csv=csv_path,
            lag=LAG,
            n_ahead=N_AHEAD,
            input_sensor=sensor_mode,
            target_sensor=sensor_mode,
            single_file_mode=True
        )
        print(f"Loaded {len(dataset)} windows from {csv_path} for sensor_mode={sensor_mode}")
        try:
            model = None
            if model_name == "patchtst":
                model = model_cls(
                    input_channels=input_size,
                    seq_len=LAG,
                    forecast_horizon=n_ahead,
                    output_size=output_size
                ).to(device)
            elif model_name == "temporal_transformer":
                model = model_cls(
                    input_size=input_size,
                    seq_len=LAG,
                    d_model=64,
                    nhead=8,
                    num_layers=3,
                    dim_feedforward=256,
                    dropout=0.1,
                    forecast_horizon=n_ahead,
                    output_size=output_size
                ).to(device)
            elif model_name == "dbn":
                model = model_cls(
                    input_size=input_size,
                    hidden_size=128,
                    num_classes=output_size,
                    n_layers=3
                ).to(device)
            elif model_name == "tcn":
                model = model_cls(
                    input_channels=input_size,
                    num_channels=[32, 32],
                    kernel_size=3,
                    dropout=0.2,
                    num_classes=output_size,
                    n_ahead=n_ahead
                ).to(device)
            elif model_name in ["lstm", "gru", "rnn"]:
                model = model_cls(
                    input_size=input_size,
                    hidden_size=256,
                    num_layers=5,
                    num_classes=output_size,
                    n_ahead=n_ahead
                ).to(device)
            elif model_name == "dlinear":
                model = model_cls(
                    seq_len=LAG,
                    forecast_horizon=n_ahead,
                    num_channels=input_size,
                    individual=False,
                    moving_avg_kernel=25
                ).to(device)
            elif model_name == "crossformer":
                model = model_cls(
                    input_channels=input_size,
                    seq_len=LAG,
                    d_model=64,
                    nhead=8,
                    num_layers=3,
                    dim_feedforward=256,
                    dropout=0.1,
                    forecast_horizon=n_ahead,
                    output_size=output_size
                ).to(device)
            elif model_name == "hybridlstmtransformer":
                model = model_cls(
                    input_size=input_size,
                    lstm_hidden_size=256,
                    lstm_layers=2,
                    d_model=64,
                    nhead=8,
                    transformer_layers=2,
                    forecast_horizon=n_ahead,
                    output_size=output_size,
                    dropout=0.1
                ).to(device)
            elif model_name == "nbeats":
                model = model_cls(
                    input_size=input_size * LAG,
                    num_stacks=2,
                    num_blocks_per_stack=3,
                    num_layers=4,
                    hidden_size=256,
                    output_size=output_size,
                    n_ahead=n_ahead,
                    stack_types=("trend", "seasonality"),
                    share_weights_in_stack=False,
                    trend_degree=2,
                    seasonality_num_harmonics=8
                ).to(device)
            elif model_name == "timeseries_transformer":
                model = model_cls(
                    input_size=input_size,
                    seq_length=LAG,
                    num_layers=3,
                    d_model=64,
                    nhead=8,
                    dim_feedforward=256,
                    dropout=0.1,
                    forecast_horizon=n_ahead,
                    output_size=output_size
                ).to(device)
            elif model_name == "informer":
                model = model_cls(
                    enc_in=input_size,
                    dec_in=input_size,
                    c_out=output_size,
                    seq_len=LAG,
                    label_len=15,
                    out_len=n_ahead,
                    d_model=64,
                    n_heads=8,
                    e_layers=2,
                    d_layers=1,
                    d_ff=256,
                    dropout=0.1
                ).to(device)
            else:
                # fallback: try input_size, hidden_size, num_layers, num_classes, n_ahead
                model = model_cls(
                    input_size=input_size,
                    hidden_size=256,
                    num_layers=5,
                    num_classes=output_size,
                    n_ahead=n_ahead
                ).to(device)
        except Exception as e:
            print(f"[ERROR] Could not instantiate model for {ckpt_path}: {e}")
            continue
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("module.", "")
                new_state_dict[new_key] = v
            model.load_state_dict(new_state_dict, strict=False)
            model.eval()
        except Exception as e:
            print(f"[ERROR] Could not load checkpoint {ckpt_path}: {e}")
            continue
        print(f"\n--- Simulating real-time inference for {model_name} ({os.path.basename(ckpt_path)}) ---")
        print(f"Using sensor_mode={sensor_mode} and input_size={input_size}")
        timings = []
        for i in range(len(dataset)):
            X, Y, *_ = dataset[i]  # Ignore action/weight
            X = X.unsqueeze(0).to(device)  # Add batch dimension
            torch.cuda.synchronize() if device == "cuda" else None
            start = time.perf_counter()
            with torch.no_grad():
                if model_name == "nbeats":
                    pred = model(X.reshape(X.size(0), -1))
                else:
                    pred = model(X)
            torch.cuda.synchronize() if device == "cuda" else None
            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000
            timings.append(elapsed_ms)
            latency_records.append({
                "model": model_name,
                "n_ahead": n_ahead,
                "latency": elapsed_ms
            })
        mean_latency = float(np.mean(timings)) if timings else float('nan')
        print(f"Model: {model_name:20s} | Mean latency: {mean_latency:.3f} ms | Windows: {len(dataset)}")
    # Compute and write mean latency per (model, n_ahead) combination
    import pandas as pd
    latencies_df = pd.DataFrame(latency_records, columns=["model", "n_ahead", "latency"])
    summary_df = latencies_df.groupby(["model", "n_ahead"], as_index=False)["latency"].mean()
    summary_df.rename(columns={"latency": "mean_latency"}, inplace=True)
    summary_df.to_csv("benchmark_inference_summary.csv", index=False)
    print("Benchmark mean latency summary saved to benchmark_inference_summary.csv")
    print("\nAll models processed.")
    return latency_records

if __name__ == "__main__":
    # benchmark_inference()
    simulate_realtime_inference_on_csv()
