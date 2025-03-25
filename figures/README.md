# Folder Naming Convention
```python
def process_file(file_path, data_dir, fs_target=148.148, emg_scale=100.0,
                 emg_cutoff=10, inertial_cutoff=5, gyro_highpass_cutoff=0.5,
                 outlier_method="mad", causal_window=5,
                 debug_std=False,
                 flag_scale=True, flag_outlier=True, flag_lowpass=False,
                 flag_moving_average=False, flag_highpass=True):
```
Above is the files used for first training

This would be renaming to processing_scale_outlier_highpass
Scaling flag is multiplying the EMG signal by the emg_scale value, if the max is over a certian value, different scaling is used
emg_cutoff/inertial_cutoff, after processing if over this value, csv file dropped
outlier_method, which outlier function to use, zscore, iqr, mad
casual window is the window size


If a scaler is added then _scaler is appened to the end