import numpy as np
import os
import re
import cv2

def find_all_npy(base_path):
    """
    Finds all .npy files in the given directory recursively.
    
    Returns:
        A sorted list of file paths based on timestamps extracted from filenames.
    """
    npy_files = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".npy"):
                npy_files.append(os.path.join(root, file))
    
    # Sort by timestamp extracted from filename
    npy_files.sort(key=lambda x: int(re.search(r"(\d+)\.npy$", os.path.basename(x)).group(1)) if re.search(r"(\d+)\.npy$", os.path.basename(x)) else 0)
    return npy_files

def load_npy(file_path):
    """
    Loads a .npy file and returns the image data.
    """
    try:
        loaded = np.load(file_path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    
    if isinstance(loaded, dict):
        return loaded.get("img_data", loaded.get("data", None))
    elif isinstance(loaded, (tuple, list)) and len(loaded) >= 3:
        return loaded[0]  # Assuming (image_data, timestamp, camera_index)
    elif isinstance(loaded, np.ndarray):
        return loaded
    else:
        print(f"Unexpected format in {file_path}")
        return None

def save_video(npy_files, output_video, fps=30):
    """
    Reads numpy images and compiles them into a video.
    """
    if not npy_files:
        print("No valid .npy files found.")
        return
    
    first_image = load_npy(npy_files[0])
    if first_image is None:
        print("Failed to load first image, aborting video creation.")
        return
    
    if first_image.dtype != np.uint8:
        first_image = (255 * (first_image - first_image.min()) / (first_image.max() - first_image.min())).astype(np.uint8)
    
    if first_image.shape[0] == 3:  # Convert (C, H, W) to (H, W, C)
        first_image = np.moveaxis(first_image, 0, -1)
    
    height, width, _ = first_image.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    for file in npy_files:
        image = load_npy(file)
        if image is None:
            continue
        
        if image.dtype != np.uint8:
            image = (255 * (image - image.min()) / (image.max() - image.min())).astype(np.uint8)
        
        if image.shape[0] == 3:
            image = np.moveaxis(image, 0, -1)  # Convert (C, H, W) to (H, W, C)
        
        out.write(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    out.release()
    print(f"Video saved as {output_video}")

# Example usage
base_dir = "/data1/dnicho26/EMG_DATASET/data/1/steps/camera_0"
npy_files = find_all_npy(base_dir)
output_video_path = "output_video.mp4"
save_video(npy_files, output_video_path)
