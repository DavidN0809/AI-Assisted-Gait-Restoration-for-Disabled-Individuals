import os
import numpy as np
import pandas as pd
import logging
from tqdm.auto import tqdm
import re

# Setup basic logging.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def find_npy_files(base_path):
    """
    Recursively finds all .npy files in the given directory.
    """
    npy_files = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".npy"):
                npy_files.append(os.path.join(root, file))
    return npy_files

def load_npy_files(base_dir):
    """
    Load npy files for machine learning training from a given base directory.
    
    This function handles:
      - Raw npy files (just an image array)
      - npy files with metadata (a dict or tuple/list containing image data, timestamp, and camera_index)
      
    If a loaded file contains a camera_index, the code checks that the file's folder name 
    (e.g., "camera_0", "camera_1", etc.) matches the embedded camera_index. If not, it updates the folder.
    
    Returns:
        A pandas DataFrame containing:
          - uuid: the UUID folder from the file path
          - action: the action folder from the file path
          - camera_folder: folder name (verified or updated)
          - camera_index: camera index (if available)
          - file_path: full path to the file
          - timestamp: timestamp from the file (if available)
          - data: the loaded image data (array)
    """
    npy_files = find_npy_files(base_dir)
    logger.info(f"Found {len(npy_files)} npy files.")
    
    records = []
    for file_path in tqdm(npy_files, desc="Processing npy files"):
        # Compute the relative path to extract UUID, action, and camera folder.
        relative_path = os.path.relpath(file_path, base_dir)
        path_parts = relative_path.split(os.sep)
        
        # We expect at least three levels: UUID/action/camera_x/<file>
        if len(path_parts) < 3:
            logger.warning(f"File {file_path} does not match expected structure UUID/action/camera_x")
            continue

        uuid = path_parts[0]
        action = path_parts[1]
        camera_folder = path_parts[2]  # e.g., "camera_0", "camera_1", etc.
        
        # Optionally extract camera index from the folder name (if present).
        camera_match = re.search(r"camera_(\d+)", camera_folder, re.IGNORECASE)
        folder_camera_index = int(camera_match.group(1)) if camera_match else None

        try:
            loaded = np.load(file_path, allow_pickle=True)
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            continue

        # Initialize variables.
        camera_index = None
        timestamp = None
        data = None

        # Check the type and structure of the loaded data.
        if isinstance(loaded, dict):
            if "camera_index" in loaded:
                # New format with metadata.
                camera_index = loaded.get("camera_index")
                # Using 'abs_time' as the timestamp key; change to "timestamp" if that’s what you use.
                timestamp = loaded.get("abs_time")
                data = loaded.get("img_data")
                # Verify folder consistency.
                if camera_index is not None:
                    expected_folder = f"camera_{camera_index}"
                    if expected_folder != camera_folder:
                        logger.info(f"File {file_path} expected in folder {expected_folder} but found in {camera_folder}. Updating folder.")
                        camera_folder = expected_folder
            else:
                # Older format stored as a dict.
                data = loaded.get("data", loaded)
                timestamp = loaded.get("timestamp")
                camera_index = None
        elif isinstance(loaded, (tuple, list)) and len(loaded) >= 3:
            data, timestamp, camera_index = loaded[0], loaded[1], loaded[2]
            if camera_index is not None:
                expected_folder = f"camera_{camera_index}"
                if expected_folder != camera_folder:
                    logger.info(f"File {file_path} expected in folder {expected_folder} but found in {camera_folder}. Updating folder.")
                    camera_folder = expected_folder
        elif isinstance(loaded, np.ndarray):
            # Raw npy array (only image data)
            data = loaded
            timestamp = None
            camera_index = None
        else:
            logger.warning(f"Unexpected data format in {file_path}")
            continue

        record = {
            "uuid": uuid,
            "action": action,
            "camera_folder": camera_folder,
            "camera_index": camera_index,
            "file_path": file_path,
            "timestamp": timestamp,
            "data": data
        }
        records.append(record)
    
    # Create a DataFrame from all collected records.
    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    # Adjust this to the parent directory that contains all the UUID directories.
    base_dir = r"D:\UNC Charlotte Dropbox\orgs-ecgr-QuantitativeImagingandAILaboratory"
    
    # Load the npy files into a DataFrame.
    df = load_npy_files(base_dir)
    print(df.head())

    # Optionally, save the DataFrame to a CSV file for further processing.
    output_csv = os.path.join(base_dir, "npy_files_data.csv")
    df.to_csv(output_csv, index=False)
    logger.info(f"DataFrame saved to {output_csv}")
