import os
import pandas as pd

def save_summary_csv(results, base_trial_dir):
    """
    Save the summary CSV with final evaluation metrics for all models.

    Parameters:
    - results: A list of dictionaries containing model metrics.
    - base_trial_dir: The base directory under which the summary CSV will be saved.

    Returns:
    - results_path: The path to the saved CSV file.
    """
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('final_val_mse')
    results_path = os.path.join(base_trial_dir, "baseline_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    return results_path
import os

def log_epoch_loss(log_file_path, epoch, train_loss, val_loss, extra_metrics):
    """
    Logs the per-epoch loss metrics into a CSV file.

    Parameters:
    - log_file_path: The path to the log file.
    - epoch: The current epoch (int).
    - train_loss: Training loss for the epoch.
    - val_loss: Validation loss for the epoch.
    - extra_metrics: A dictionary containing additional metrics (e.g., rmse, mae, r2_score, corr_coef).
    """
    header = "epoch,train_loss,val_loss,rmse,mae,r2_score,corr_coef\n"
    # If the log file does not exist, create it with a header.
    if not os.path.exists(log_file_path):
        with open(log_file_path, "w") as f:
            f.write(header)
    # Append the current epoch metrics.
    with open(log_file_path, "a") as f:
        f.write(
            f"{epoch},{train_loss},{val_loss},"
            f"{extra_metrics.get('rmse', 'NA')},{extra_metrics.get('mae', 'NA')},"
            f"{extra_metrics.get('r2_score', 'NA')},{extra_metrics.get('corr_coef', 'NA')}\n"
        )
