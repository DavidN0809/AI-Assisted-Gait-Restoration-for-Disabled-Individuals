#scripts folder
used to check data, checks for all 6 sensors etc
checks video frames to make sure all camera recorded entire duration

preprocessing.py
takes raw data, and created 3 dirs
one is raw data with sensor mapping fixed, ie if no senso 5 but a sensor 7, rename 7 to 5

the next resamples the data to the lowest hz or user specified hz

the final applies a butterworth filter ontop of that