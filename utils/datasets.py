import os
import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import concurrent.futures
import logging

# Set up logging.
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def process_csv(file_path):
    """
    Reads the CSV file and returns a numpy array with sensor columns re-ordered.
    It performs the following:
      1. Drops columns whose names are NaN or that contain "time" (case-insensitive).
      2. Drops columns whose names contain "IMP" (case-insensitive).
      3. Groups columns by sensor number (using regex on the column names, e.g. 'sensor 0' or 'sensor_0').
      4. Verifies that sensor groups 0 to 5 exist.
      5. Reorders the columns so that sensor 0 columns come first, then sensor 1, etc.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Sensor CSV file not found: {file_path}")
        
    df = pd.read_csv(file_path)
    
    # Drop columns with NaN names or that contain "time"
    drop_cols = [col for col in df.columns 
                 if pd.isna(col) or col.strip().lower() == "nan" or "time" in col.lower()]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    
    # Drop columns that contain "IMP" (case-insensitive)
    df = df[[col for col in df.columns if "imp" not in col.lower()]]
    
    # Group columns by sensor number using regex.
    groups = {}
    for col in df.columns:
        m = re.search(r'sensor[_\s]*(\d+)', col, re.IGNORECASE)
        if m:
            sensor_num = int(m.group(1))
            groups.setdefault(sensor_num, []).append(col)
    # Ensure sensors 0 to 5 are present.
    for i in range(6):
        if i not in groups:
            raise ValueError(f"File '{file_path}' is missing sensor group {i}")
    
    # For each sensor group, sort columns by their original order.
    new_order = []
    for i in range(6):
        # Sort the columns in group i according to their order in the dataframe.
        group_cols = groups[i]
        group_cols_sorted = sorted(group_cols, key=lambda x: df.columns.get_loc(x))
        new_order.extend(group_cols_sorted)
    
    # Reorder the dataframe.
    df = df[new_order]
    
    # Select only numeric columns (in case non-numeric columns remain).
    df = df.select_dtypes(include=[np.number])
    
    sensor_data = df.to_numpy()
    n_features = sensor_data.shape[1]
    return sensor_data, n_features

class EMG_dataset(Dataset):
    """
    A PyTorch Dataset for loading EMG sensor data from multiple CSV files.

    For each CSV file:
      1. Drops any columns with NaN names or containing "time" (e.g., a time column).
      2. Drops any columns whose names contain "IMP".
      3. Scans the remaining column names for sensor identifiers (e.g. "sensor 0", "sensor_1", etc.)
         and reorders the columns so that sensor 0 comes first, then sensor 1, etc.
      4. Checks that sensor groups 0 through 5 are present.
      
    A sliding window is then created:
      - The first 'lag' rows are used as the history window.
      - The following 'n_ahead' rows are used as the forecast window.

    Finally, the sensor data in each window is split along the feature dimension:
      - If input_leg=="right": the input (X) is the right leg data (sensors 3–5) and the target (Y)
        is the left leg data (sensors 0–2).
      - If input_leg=="left": the input (X) is the left leg data and the target (Y) is the right leg data.

    Args:
        index_file (str): Path to the index CSV file (should contain a column with file paths).
        lag (int): Number of time steps to use as input (history window).
        n_ahead (int): Number of time steps to forecast.
        path_column (str): Name of the column in the index CSV that holds sensor CSV file paths.
        expected_features (int, optional): If provided, each sensor CSV will be forced to have this
                                           number of numeric columns (after processing). If None, the
                                           maximum among files is used.
        input_leg (str): Which leg to use as the input. Must be either "left" or "right". The other leg
                         will be used as ground truth.
    """
    def __init__(self, index_file, lag, n_ahead, path_column="file_path", expected_features=None, input_leg="right"):
        if input_leg not in ["left", "right"]:
            raise ValueError("input_leg must be 'left' or 'right'")
        self.input_leg = input_leg
        self.lag = lag
        self.n_ahead = n_ahead
        
        self.index_df = pd.read_csv(index_file)
        self.file_paths = self.index_df[path_column].tolist()
        logger.info(f"Loaded {len(self.file_paths)} file paths from index.")
        self.data = []      # list to hold sensor data arrays
        self.indices = []   # list to map a global index to (file_idx, start_idx)
        
        # Phase 1: Determine expected number of features (if not provided).
        if expected_features is None:
            def get_features(file_path):
                sensor_data, n_features = process_csv(file_path)
                logger.debug(f"File '{file_path}' has {n_features} numeric features after processing.")
                # Check that n_features is divisible by 6 (i.e. 6 sensor groups).
                if n_features % 6 != 0:
                    raise ValueError(f"File '{file_path}' does not have a number of sensor columns divisible by 6 (got {n_features}).")
                return n_features
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                features_list = list(executor.map(get_features, self.file_paths))
            self.expected_features = max(features_list)
            logger.info(f"Determined expected features (max across files): {self.expected_features}")
        else:
            if expected_features % 6 != 0:
                raise ValueError("expected_features must be divisible by 6.")
            self.expected_features = expected_features
            logger.info(f"Using provided expected_features: {self.expected_features}")
        
        # Phase 2: Process each sensor CSV file concurrently.
        def process_file(args):
            file_idx, file_path = args
            logger.info(f"Processing file {file_idx}: {file_path}")
            sensor_data, current_features = process_csv(file_path)
            if current_features % 6 != 0:
                raise ValueError(f"File '{file_path}' does not have a number of sensor columns divisible by 6 (got {current_features}).")
            
            # Pad or truncate sensor_data to match expected_features.
            if current_features < self.expected_features:
                sensor_data = np.pad(sensor_data, ((0, 0), (0, self.expected_features - current_features)), mode='constant')
                logger.debug(f"Padded file {file_path} from {current_features} to {self.expected_features} features.")
            elif current_features > self.expected_features:
                sensor_data = sensor_data[:, :self.expected_features]
                logger.debug(f"Truncated file {file_path} from {current_features} to {self.expected_features} features.")
            
            num_rows = sensor_data.shape[0]
            num_windows = num_rows - (lag + n_ahead) + 1
            indices_for_file = []
            if num_windows > 0:
                indices_for_file = list(range(num_windows))
                logger.info(f"File {file_path} processed: {num_windows} windows created.")
            else:
                logger.warning(f"File {file_path} does not have enough rows (required: {lag+n_ahead}, got: {num_rows}).")
            return file_idx, sensor_data, indices_for_file
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_file, enumerate(self.file_paths)))
        
        # Preserve original file order.
        results.sort(key=lambda x: x[0])
        for file_idx, sensor_data, indices_for_file in results:
            self.data.append(sensor_data)
            for start_idx in indices_for_file:
                self.indices.append((file_idx, start_idx))
        
        if len(self.indices) == 0:
            raise ValueError("No valid windows could be extracted from the data. Check your CSV files and parameters.")
        logger.info(f"Initialization complete: {len(self.indices)} sliding windows created across {len(self.data)} files.")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        file_idx, start_idx = self.indices[idx]
        sensor_data = self.data[file_idx]
        
        # Extract sliding windows for history and forecast.
        X_full = sensor_data[start_idx : start_idx + self.lag, :]
        Y_full = sensor_data[start_idx + self.lag : start_idx + self.lag + self.n_ahead, :]
        
        total_features = X_full.shape[1]
        if total_features % 6 != 0:
            raise ValueError("The number of sensor columns is not divisible by 6.")
        group_size = total_features // 6
        
        # Define slices for left leg (sensor groups 0-2) and right leg (sensor groups 3-5)
        left_slice = slice(0, 3 * group_size)
        right_slice = slice(3 * group_size, 6 * group_size)
        
        # Split the history and forecast windows.
        X_left = X_full[:, left_slice]
        X_right = X_full[:, right_slice]
        Y_left = Y_full[:, left_slice]
        Y_right = Y_full[:, right_slice]
        
        # Choose which leg is the input and which is the ground truth.
        if self.input_leg == "right":
            X = X_right  # input from right leg
            Y = Y_left   # target from left leg
        else:  # self.input_leg == "left"
            X = X_left   # input from left leg
            Y = Y_right  # target from right leg
        
        # Convert to torch tensors.
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)
        return X, Y

class PoseDataset(Dataset):
    """
    A PyTorch Dataset for loading camera image data from a CSV file (generated by your npy loader)
    and extracting pose keypoints using MediaPipe.

    Each row in the CSV is expected to contain at least:
      - file_path: the full path to the npy file or image file
      - timestamp: timestamp information (if available)
      - camera_index: the index of the camera (if available)
      - (optionally) additional metadata like uuid or action
    """
    def __init__(self, csv_file, transform=None, static_image_mode=True, min_detection_confidence=0.5):
        """
        Args:
            csv_file (str): Path to the CSV file that holds camera data.
            transform (callable, optional): Optional transform to be applied to the loaded image.
            static_image_mode (bool): Whether to treat images as static. Set True for independent image processing.
            min_detection_confidence (float): Minimum confidence for pose detection.
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        
        # Initialize MediaPipe Pose.
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=static_image_mode, 
                                      min_detection_confidence=min_detection_confidence)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row["file_path"]
        
        # Load image using OpenCV.
        image = cv2.imread(file_path)
        if image is None:
            logger.error(f"Could not read image from {file_path}")
            sample = {"keypoints": None,
                      "timestamp": row.get("timestamp", None),
                      "camera_index": row.get("camera_index", None),
                      "uuid": row.get("uuid", None),
                      "action": row.get("action", None)}
            return sample

        # Optionally apply any transformations.
        if self.transform:
            image = self.transform(image)
        
        # Convert image from BGR to RGB (as required by MediaPipe).
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        keypoints = []
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                # Each keypoint: (x, y, visibility)
                keypoints.append([landmark.x, landmark.y, landmark.visibility])
        else:
            logger.warning(f"No pose landmarks detected for image {file_path}")
        
        # Convert list of keypoints to a numpy array.
        keypoints = np.array(keypoints)
        
        sample = {
            "keypoints": keypoints,
            "timestamp": row.get("timestamp", None),
            "camera_index": row.get("camera_index", None),
            "uuid": row.get("uuid", None),
            "action": row.get("action", None)
        }
        return sample