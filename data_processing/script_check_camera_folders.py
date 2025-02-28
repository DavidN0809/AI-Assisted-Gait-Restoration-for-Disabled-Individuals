  GNU nano 4.8                                      script_check_cams.py                                                import os

def check_empty_or_missing_cameras(base_path):
    actions = ["steps", "treadmill", "walk and turn left", "walk and turn right"]
    cameras = ["camera_0", "camera_2", "camera_3"]

    for uuid_dir in os.listdir(base_path):
        uuid_path = os.path.join(base_path, uuid_dir)
        if os.path.isdir(uuid_path):
            for action in actions:
                action_path = os.path.join(uuid_path, action)
                if os.path.isdir(action_path):
                    missing_or_empty = []
                    for camera in cameras:
                        camera_path = os.path.join(action_path, camera)
                        if not os.path.isdir(camera_path) or len(os.listdir(camera_path)) == 0:
                            missing_or_empty.append(camera)

                    if missing_or_empty:
                        print(f"UUID: {uuid_dir}, action: {action} - missing or empty camera(s): {', '.join(missing_or_>

check_empty_or_missing_cameras("data")
