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
    """Recursively finds npy files in the directory structure."""
    npy_files = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".npy"):
                npy_files.append(os.path.join(root, file))
    return npy_files

# Adjust this to the parent directory that contains all the UUID directories.
base_dir = "D:\UNC Charlotte Dropbox\orgs-ecgr-QuantitativeImagingandAILaboratory"

# Find all npy files.
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
    
    # Optionally extract camera index from folder name (using regex) if needed.
    camera_match = re.search(r"camera_(\d+)", camera_folder, re.IGNORECASE)
    camera_index = int(camera_match.group(1)) if camera_match else None

    try:
        loaded = np.load(file_path, allow_pickle=True)
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        continue
    
    # Determine file content format.
    if isinstance(loaded, dict):
        data = loaded.get("data")
        timestamp = loaded.get("timestamp")
        camera = loaded.get("camera")
    elif isinstance(loaded, (tuple, list)) and len(loaded) >= 3:
        data, timestamp, camera = loaded[0], loaded[1], loaded[2]
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
        "data": data,
        "camera": camera
    }
    records.append(record)

# Create a DataFrame from all collected records.
df = pd.DataFrame(records)
print(df.head())
