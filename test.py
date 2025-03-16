import time
import torch
from datetime import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt

from models.models import BasicLSTM
from utils.datasets import EMG_dataset

def extract_value(values, upper_leg, lower_leg):
    # Scale the values based on the leg parameters
    values[:2] *= upper_leg[0]
    values[2:] *= lower_leg[0]
    
    # Safety measures: clip values to the maximum allowed
    values[:2] = np.clip(values[:2], 0, upper_leg[1])
    values[2:] = np.clip(values[2:], 0, lower_leg[1])
    
    return [int(v) for v in values]

def color_map(x):
    # Generate a color based on the normalized value x (between 0 and 1)
    r = max(-np.e**(4*x-4)+1, 0)
    g = min(np.e**(8*x-4), 1)
    b = 0
    return (int(b*255), int(g*255), int(r*255))

def draw_leg(leg_img, circles, values, upper_leg, lower_leg):
    # Draw circles on the image representing leg parts
    for c, v in zip(circles, values):
        # Determine maximum value for current leg part
        d = upper_leg[1] if v in values[:2] else lower_leg[1]
        color = color_map(v/d)
        
        # Draw the filled circle and a border circle
        leg_img = cv2.circle(leg_img, c, 30, color, -1)
        leg_img = cv2.circle(leg_img, c, 33, (0, 0, 0), 5)
        
    return leg_img

# Leg coordinates for left and right legs
ham_l_circle = (65, 200)
quad_l_circle = (220, 200)
anti_l_circle = (50, 500)
calve_l_circle = (185, 500)
ham_r_circle = (320, 200)
quad_r_circle = (490, 200)
anti_r_circle = (320, 500)
calve_r_circle = (455, 500)

circles_left = [quad_l_circle, ham_l_circle, calve_l_circle, anti_l_circle]
circles_right = [quad_r_circle, ham_r_circle, calve_r_circle, anti_r_circle]

# Leg parameter definitions: [scaling factor, maximum value]
upper_leg = [120, 65]
lower_leg = [160, 120]

# Video output settings
FPS = 30
leg_img_template = cv2.imread("figures/legs.png")
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video_writer = cv2.VideoWriter("legs.mp4", fourcc, FPS, (leg_img_template.shape[1], leg_img_template.shape[0]))

# Load dataset and model
csv_path = "/data1/dnicho26/EMG_DATASET/data/1/steps/1738797999.1432617.csv"
# Note: Here EMGO_dataset should be set up to return all signals.
# For the “all” configuration, the input size is 21.
dataset = EMG_dataset(csv_path, lag=24, n_ahead=12)

input_size = 21       # Using all available signals
hidden_size = 128
num_layers = 5
output_size = dataset.n_ahead

model = BasicLSTM(input_size, hidden_size, num_layers, output_size)
# Load the model saved for the "all" configuration (adjust the path as needed)
model.load_state_dict(torch.load('models/trained/model_all.pt'))
model.eval()  # Set model to evaluation mode

# Process each sample in the dataset and write frames to video
for i in range(len(dataset)):
    
    initial_frame_time = datetime.now()
    
    # Retrieve data sample from dataset
    # Note: The dataset should provide the full input signals.
    emg_accel, right_emg = dataset[i]
    # For demonstration, assume right_emg and left_emg are derived as follows.
    # (Adjust indexing if your dataset structure differs.)
    right_emg = [val.item() for val in right_emg[0, :]]
    left_emg = [val.item() for val in emg_accel[-1, :]]
    
    # Get model prediction
    with torch.no_grad():
        # Ensure input shape matches the model’s expectation (batch_size, sequence_length, input_size)
        pred_emg = [val.item() for val in model(emg_accel.view(1, 24, input_size))[0, 0, :]]
    
    # Process values for drawing
    pred_emg = extract_value(np.array(pred_emg), upper_leg, lower_leg)
    right_emg = extract_value(np.array(right_emg), upper_leg, lower_leg)
    left_emg = extract_value(np.array(left_emg), upper_leg, lower_leg)
    print(f"Frame {i}: Predicted EMG values: {pred_emg}")
    
    # Create a fresh frame from the template for each iteration
    frame = leg_img_template.copy()
    frame = draw_leg(frame, circles_left, left_emg, upper_leg, lower_leg)
    frame = draw_leg(frame, circles_right, right_emg, upper_leg, lower_leg)
    
    # Write the frame to the video file
    video_writer.write(frame)
    
    # Optionally display the frame
    cv2.imshow("Legs", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Maintain timing to achieve the desired FPS
    while (datetime.now() - initial_frame_time).total_seconds() < (1 / FPS):
        pass

video_writer.release()
cv2.destroyAllWindows()
