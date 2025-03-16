import serial
import time
import torch
from datetime import datetime
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

import cv2
import numpy as np
import matplotlib.pyplot as plt

from models.models import BasicLSTM
from utils.datasets import EMG_dataset

def send_data(ser, pair):
    
    ser.flushInput()
    data = bytes(','.join(str(i) for i in pair) + '\n', 'utf-8')
    ser.write(data)
    
def extract_value(values, upper_leg, lower_leg):
    values[:2] *= upper_leg[0]
    values[2:] *= lower_leg[0]
    
    # Safety Measures
    values[:2] = np.clip(values[:2], 0, upper_leg[1])
    values[2:] = np.clip(values[2:], 0, lower_leg[1])
    
    return [int(v) for v in values]

def color_map(x):
    r = max(-np.e**(4*x-4)+1, 0)
    g = min(np.e**(8*x-4), 1)
    b = 0
    
    return (int(b*255),int(g*255),int(r*255))

def draw_leg(leg_img, circles, values):
    for c, v in zip(circles, values):
        
        d = upper_leg[1] if v in values[:2] else lower_leg[1]
        color = color_map(v/d)
        
        leg_img = cv2.circle(leg_img, c, 30, color, -1)
        leg_img = cv2.circle(leg_img, c, 33, BLACK, 5)
        
    return leg_img

#Leg Cords

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

BLACK = (0,0,0)

# ser = serial.Serial('COM6', 9600)

csv_path = "emg.csv"
dataset = EMG_dataset(csv_path, lag=24, n_ahead=12)


input_size = 4
hidden_size = 128
num_layers = 5
output_size = dataset.n_ahead

model = BasicLSTM(input_size, hidden_size, num_layers, output_size)
model.load_state_dict(torch.load('models/pre-trained/EMG_Last.pt'))

leg_img = cv2.imread("figures/legs.png")
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

upper_leg = [120, 65]
lower_leg = [160, 120]
FPS = 30

legs_out = cv2.VideoWriter(f"legs.mp4", fourcc, FPS, (leg_img.shape[1], leg_img.shape[0]))

for i in range(dataset.__len__()):
    
    initialFrameTime = datetime.now()
    
    emg_accel, right_emg = dataset.__getitem__(i)
    right_emg = [i.item() for i in right_emg[0, :]]
    left_emg = [i.item() for i in emg_accel[-1, :]]
    
    with torch.no_grad():
        pred_emg = [i.item() for i in model(emg_accel.view(1, 24, 4))[0, 0, :] ]
    
    
    pred_emg = extract_value(np.array(pred_emg), upper_leg, lower_leg)
    right_emg = extract_value(np.array(right_emg), upper_leg, lower_leg)
    left_emg = extract_value(np.array(left_emg), upper_leg, lower_leg)
    print(pred_emg)
    
    # send_data(ser, pred_emg)
    
    leg_img = draw_leg(leg_img, circles_left, left_emg)
    leg_img = draw_leg(leg_img, circles_right, right_emg)
    
    legs_out.write(leg_img)
    
    cv2.imshow("leg", leg_img)
    cv2.waitKey(1)
    
    
    lateFrameDifference = float( str(datetime.now() - initialFrameTime)[7:] )
    
    while (lateFrameDifference < (1/FPS)):
        lateFrameDifference = float( str(datetime.now() - initialFrameTime)[7:] )
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # sds = input()
    
legs_out.release()
# ser.close()

# send_data(ser, [0,0,0,0])
# ser.close()

# ser = serial.Serial('COM4', 9600)  # Replace COM4 with the port name of your Arduino
# c = 0

# while True:
    
    
    
    
#     value = 30  # Replace 42 with the integer you want to send
#     ser.flushInput()  # Clear the input buffer
#     ser.write(value.to_bytes(2, 'big'))  # Convert the integer to bytes and send it
    
#     time.sleep(2)  # Wait for 1 second before sending the next value
    
    
#     value = 0  # Replace 42 with the integer you want to send
#     ser.flushInput()  # Clear the input buffer
#     ser.write(value.to_bytes(2, 'big'))  # Convert the integer to bytes and send it
    
#     time.sleep(2)
    
#     c+=1
    
    

# emg_path = "M:/Datasets/shock_walk/processed/2.0mph.csv"
# emg_data = pd.read_csv(emg_path)
# display(emg_data)