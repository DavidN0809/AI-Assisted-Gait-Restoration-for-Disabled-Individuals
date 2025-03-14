
import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import logging
from torch.utils.data import Dataset
from utils.datasets import PoseDataset

# Set up logging.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Example usage:
if __name__ == "__main__":
    # Path to the CSV file generated from your npy loader.
    csv_path = r"D:\UNC Charlotte Dropbox\orgs-ecgr-QuantitativeImagingandAILaboratory\npy_files_data.csv"
    
    # Create the dataset instance.
    pose_dataset = PoseDataset(csv_path)
    
    # Retrieve an example sample.
    sample = pose_dataset[0]
    print("Keypoints shape:", sample["keypoints"].shape)
    print("Metadata:", {"timestamp": sample["timestamp"], "camera_index": sample["camera_index"]})
