import os

# Path to the 'data' directory
data_dir = 'data'

# List of action folder names to check
action_folders = ['steps', 'treadmill', 'walk and turn left', 'walk and turn right']

# Camera folders that need to exist
required_cameras = ['camera_0', 'camera_2', 'camera_3']

def check_folder_empty_or_missing_cameras(folder_uuid, action):
    for action_folder in action_folders:
        action_folder_path = os.path.join(data_dir, folder_uuid, action_folder)
        if os.path.isdir(action_folder_path):  # Check if the action folder exists
            # Check the camera folders inside the action folder
            missing_cameras = []
            for camera in required_cameras:
                camera_folder_path = os.path.join(action_folder_path, camera)
                if not os.path.exists(camera_folder_path) or not os.listdir(camera_folder_path):
                    missing_cameras.append(camera)

            if missing_cameras:
                print(f"UUID: {folder_uuid}, Action: {action_folder} - Missing or empty cameras: {', '.join(missing_cameras)}")
        else:
            print(f"UUID: {folder_uuid}, Action: {action_folder} - Action folder is missing")

def check_folders():
    for folder_uuid in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_uuid)
        if os.path.isdir(folder_path) and folder_uuid.isdigit():
            for action in action_folders:
                check_folder_empty_or_missing_cameras(folder_uuid, action)

if __name__ == '__main__':
    check_folders()
