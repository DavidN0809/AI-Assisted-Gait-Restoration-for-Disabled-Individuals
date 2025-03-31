import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss, NBeats
from pytorch_forecasting.data import GroupNormalizer
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import dask.dataframe as dd

# If long-format Parquet files exist, load them. Otherwise, process the original datasets.
from dask.diagnostics import ProgressBar
import os
import time
# Append parent directory to locate modules (assumes a project folder structure)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.datasets import EMGTimeDataset  # Use the new dataset class that leverages CSV time info

def convert_dataset_to_long_format(dataset, group_col_name="group_id", 
                                   target_col_name="target", target_channel=0):
    """
    Converts a dataset loaded via EMGTimeDataset into a long-format DataFrame.
    For each sample (window) the full sensor window is used to extract the target series.
    This version generates a proper time index for each sample window.
    """
    # Check if randomization is off:
    if not dataset.randomize_legs:
        target_leg = "right"
        candidate_target_indices = list(range(21, 42))
        if dataset.target_sensor != "all":
            target_cols = [idx for idx in candidate_target_indices 
                           if dataset.sensor_columns[idx].lower().find(dataset.target_sensor) != -1]
            if not target_cols:
                raise ValueError(f"No target columns found for sensor type '{dataset.target_sensor}' in {target_leg} leg.")
        else:
            target_cols = candidate_target_indices

        df_list = []
        for i in tqdm(range(len(dataset)), desc="Processing samples"):
            window_sensor, window_time, action = dataset.samples[i]
            full_series = window_sensor[:, target_cols]
            target = full_series[:, target_channel]
            # Determine number of time steps for this window. Ideally this should be lag+n_ahead.
            n_steps = len(window_time)
            # If window_time is constant (i.e. all values are the same), generate a sequence:
            if np.all(window_time == window_time[0]):
                sample_time = np.arange(window_time[0], window_time[0] + n_steps)
            else:
                sample_time = window_time  # use the provided varying time vector
            temp_df = pd.DataFrame({
                "time_idx": sample_time,
                target_col_name: target
            })
            temp_df[group_col_name] = i
            temp_df["action"] = action
            temp_df["target_leg"] = target_leg
            df_list.append(temp_df)
        df_combined = pd.concat(df_list, ignore_index=True)
        print(f"Converted DataFrame shape: {df_combined.shape}")
        return df_combined
    else:
        # If randomization is enabled, keep your existing threaded approach (apply similar fix inside process_sample)
        def process_sample(i):
            window_sensor, window_time, action = dataset.samples[i]
            _, _, _, _, _, target_leg = dataset[i]
            if target_leg == "left":
                candidate_target_indices = list(range(0, 21))
            else:
                candidate_target_indices = list(range(21, 42))
            if dataset.target_sensor != "all":
                target_cols = [idx for idx in candidate_target_indices 
                               if dataset.sensor_columns[idx].lower().find(dataset.target_sensor) != -1]
                if not target_cols:
                    raise ValueError(f"No target columns found for sensor type '{dataset.target_sensor}' in {target_leg} leg.")
            else:
                target_cols = candidate_target_indices

            full_series = window_sensor[:, target_cols]
            target = full_series[:, target_channel]
            n_steps = len(window_time)
            # Use a generated time sequence if window_time is constant:
            window_time = np.array(window_time)
            if np.all(window_time == window_time[0]):
                sample_time = np.arange(window_time[0], window_time[0] + n_steps)
            else:
                sample_time = window_time

            if len(np.unique(window_time)) == 1:
                sample_time = np.arange(window_time[0], window_time[0] + n_steps)
            else:
                sample_time = window_time

            sample_records = []
            for t_val, val in zip(sample_time, target):
                sample_records.append({
                    group_col_name: i,
                    "time_idx": t_val,
                    target_col_name: val,
                    "action": action,
                    "target_leg": target_leg
                })
            return sample_records

        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(process_sample, range(len(dataset))),
                total=len(dataset),
                desc="Processing samples"
            ))
        records = [record for sublist in results for record in sublist]
        df_combined = pd.DataFrame(records)
        print(f"Converted DataFrame shape: {df_combined.shape}")
        return df_combined


###############################################################################
# CONFIGURATION & DATASET LOADING FOR TRAIN, VAL, TEST
###############################################################################
base_dir_dataset = "/data1/dnicho26/EMG_DATASET/data/final-proc-server/index_files"

train_index_csv = os.path.join(base_dir_dataset, "index_train.csv")
val_index_csv   = os.path.join(base_dir_dataset, "index_val.csv")
test_index_csv  = os.path.join(base_dir_dataset, "index_test.csv")

lag = 60       # historical window length (number of time steps)
n_ahead = 5    # forecast horizon (number of future time steps)
input_sensor = "all"   # could be "all", "emg", "acc", "gyro"
target_sensor = "emg"  # target sensor type; this will typically extract 3 channels
randomize_legs = False

print("Started loading datasets\n")

# Define paths for the long-format Parquet files.
long_train_parquet = os.path.join(base_dir_dataset, "long_format_train.parquet")
long_val_parquet   = os.path.join(base_dir_dataset, "long_format_val.parquet")
long_test_parquet  = os.path.join(base_dir_dataset, "long_format_test.parquet")

# if os.path.exists(long_train_parquet) and os.path.exists(long_val_parquet) and os.path.exists(long_test_parquet):
#     print("Long-format Parquet files found. Loading datasets from Parquet files...\n")
#     train_df = pd.read_parquet(long_train_parquet)
#     val_df   = pd.read_parquet(long_val_parquet)
#     test_df  = pd.read_parquet(long_test_parquet)
# else:
print("Long-format Parquet files not found. Processing datasets...\n")
# Create the dataset objects.
train_dataset = EMGTimeDataset(train_index_csv, lag, n_ahead, input_sensor, target_sensor, randomize_legs)
val_dataset   = EMGTimeDataset(val_index_csv, lag, n_ahead, input_sensor, target_sensor, randomize_legs)
test_dataset  = EMGTimeDataset(test_index_csv, lag, n_ahead, input_sensor, target_sensor, randomize_legs)

def process_conversion(name, dataset):
    start = time.time()
    print(f"Processing {name} dataset")
    df = convert_dataset_to_long_format(dataset)
    elapsed = time.time() - start
    print(f"Finished {name} dataset in {elapsed:.2f} seconds")
    return name, df

results = {}
max_workers = 32  
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {
        executor.submit(process_conversion, "train", train_dataset): "train",
        executor.submit(process_conversion, "val", val_dataset): "val",
        executor.submit(process_conversion, "test", test_dataset): "test",
    }
    for future in as_completed(futures):
        name, df = future.result()
        results[name] = df

train_df = results.get("train")
val_df   = results.get("val")
test_df  = results.get("test")

print("Finished processing datasets\n")

# Save long-format DataFrames to Parquet for future use.
train_df.to_parquet(long_train_parquet, index=False)
val_df.to_parquet(long_val_parquet, index=False)
test_df.to_parquet(long_test_parquet, index=False)
print("Saved long-format Parquet files for train, val, and test.\n")

###############################################################################
# FORECASTING DATASET CREATION & MODEL TRAINING
###############################################################################
max_encoder_length = lag       # historical window length
max_prediction_length = n_ahead
# Multiply time_idx values by 1e6 to preserve microsecond-level differences
scale_factor = 1e6
# --- After converting/saving the DataFrames ---

# Print dtype and basic stats
print(train_df["time_idx"].dtype)
print("Min time_idx:", train_df["time_idx"].min())
print("Max time_idx:", train_df["time_idx"].max())

unique_times = np.sort(train_df["time_idx"].unique())
if len(unique_times) > 1:
    time_step = unique_times[1] - unique_times[0]
else:
    time_step = 1
print("Estimated time step between samples:", time_step)

# Adjust training cutoff: subtract the equivalent of max_prediction_length time steps
training_cutoff = train_df["time_idx"].max() - (max_prediction_length * time_step)
print("Adjusted Training cutoff:", training_cutoff)

# Print number of groups (each sample gets its own group in your conversion)
n_groups = train_df["group_id"].nunique()
print("Number of groups:", n_groups)

# Optionally, print a few stats per group (e.g., min and max time_idx per group)
group_stats = train_df.groupby("group_id")["time_idx"].agg(["min", "max", "count"]).head(10)
print("Group stats (first 10 groups):")
print(group_stats)

# Now filter the DataFrame using the adjusted training_cutoff
train_df_filtered = train_df[train_df.time_idx <= training_cutoff]
n_train_samples = len(train_df_filtered)
print("Number of training samples after filtering:", n_train_samples)

# Ensure the time_idx column is integer type
train_df_filtered = train_df_filtered.copy()
if not np.issubdtype(train_df_filtered["time_idx"].dtype, np.integer):
    train_df_filtered["time_idx"] = train_df_filtered["time_idx"].astype(int)


# Construct the TimeSeriesDataSet using the filtered DataFrame.
# (Note: allow_missing_timesteps=True is kept)
training_ts = TimeSeriesDataSet(
    train_df_filtered,
    time_idx="time_idx",
    target="target",
    group_ids=["group_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["target"],
    target_normalizer=GroupNormalizer(groups=["group_id"], transformation="softplus"),
    allow_missing_timesteps=True
)

# If the warning about groups being removed appears, check the stats above.
print("Creating validation dataset from training_ts...")
validation_ts = TimeSeriesDataSet.from_dataset(training_ts, val_df, predict=True, stop_randomization=True)

train_dataloader = DataLoader(training_ts, batch_size=128, shuffle=True, num_workers=4)
val_dataloader   = DataLoader(validation_ts, batch_size=128, shuffle=False, num_workers=4)
print("start training")
# Option 1: Train a TemporalFusionTransformer.
model = TemporalFusionTransformer.from_dataset(
    training_ts,
    learning_rate=1e-3,
    hidden_size=128,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=16,
    output_size=7,  # typically the number of quantiles; adjust if needed
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)

# Option 2: Train a NBeats model (uncomment to use).
# model = NBeats.from_dataset(
#     training_ts,
#     learning_rate=1e-3,
#     hidden_layer_units=128,
#     dropout=0.1,
#     loss=QuantileLoss(),
# )

trainer = Trainer(
    max_epochs=50,
    gpus=1 if torch.cuda.is_available() else 0,
)

trainer.fit(model, train_dataloader, val_dataloader)

raw_predictions, x = model.predict(val_dataloader, mode="raw", return_x=True)
pred = raw_predictions[0]["prediction"]
true = x["decoder_target"][0]
encoder_target = x["encoder_target"][0]

combined_true = np.concatenate([encoder_target[:, 0], true[:, 0]])
combined_pred = np.concatenate([encoder_target[:, 0], pred[:, 0]])

plt.figure(figsize=(10,6))
plt.plot(combined_true, label="Ground Truth", marker="o")
plt.plot(combined_pred, label="Prediction", marker="x", linestyle="--")
plt.axvline(x=len(encoder_target)-0.5, color="black", linestyle="--", label="Forecast Boundary")
plt.xlabel("Time Step")
plt.ylabel("Target")
plt.title("Forecasting Prediction vs Ground Truth")
plt.legend()
forecast_plot_path = "./forecasting_prediction.png"
plt.savefig(forecast_plot_path)
plt.close()
print(f"Saved forecasting prediction plot to {forecast_plot_path}")
