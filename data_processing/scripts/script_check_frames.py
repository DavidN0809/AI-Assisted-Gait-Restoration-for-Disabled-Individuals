import os

def get_action_folders(user_folder):
    """
    Finds all action folders under a given user folder.
    """
    action_folders = []
    for action in os.listdir(user_folder):
        action_folder = os.path.join(user_folder, action)
        if os.path.isdir(action_folder):
            action_folders.append(action)
    return action_folders

def check_camera_frames_for_uuid(user_folder):
    """
    For a given user folder (UUID), detects actions dynamically,
    and counts the number of files in each camera folder.
    Returns mismatched actions with their frame counts if they don't match,
    otherwise returns 'UUID/<action> is matched'.
    """
    results = []
    action_folders = get_action_folders(user_folder)
    
    for action in action_folders:
        action_folder = os.path.join(user_folder, action)
        if os.path.isdir(action_folder):
            camera_frame_counts = []
            for camera in os.listdir(action_folder):
                camera_folder = os.path.join(action_folder, camera)
                if camera.startswith('camera_') and os.path.isdir(camera_folder):
                    num_files = len([
                        f for f in os.listdir(camera_folder)
                        if os.path.isfile(os.path.join(camera_folder, f))
                    ])
                    camera_frame_counts.append(num_files)
            
            # Check if all camera folders for this action have the same frame count
            if len(set(camera_frame_counts)) == 1:
                results.append(f"{os.path.basename(user_folder)}/{action} is matched")
            else:
                results.append(f"{os.path.basename(user_folder)}/{action} mismatch: {camera_frame_counts}")
    
    return results

if __name__ == "__main__":
    uuid_folder = r'D:\UNC Charlotte Dropbox\orgs-ecgr-QuantitativeImagingandAILaboratory\8'
    print("Checking camera frames for the specified UUID...")
    mismatches = check_camera_frames_for_uuid(uuid_folder)
    for mismatch in mismatches:
        print(mismatch)
