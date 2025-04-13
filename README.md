# AI-Assisted-Gait-Restoration-for-Disabled-Individuals
```
conda activate EMG
screen -dmS training-acc bash -c "source activate EMG; python training-acc.py"
```

# Sync data from dropbox
Uses rclone
setup a remote via
```
rclone config
```
select new remove via n
create a name, I used dropbox, if you use a different name modify the script
then enter 13 for dropbox
press enter twice for oath to be left blank
then press enter twice again to use auto configs

run the followin to get the IP of the server
```
ip a
```
then run this command on your local machine, replace "10.16.30.120" with the ip

```
 ssh -L 53682:127.0.0.1:53682 dnicho26@10.16.30.120
```
then copy the link from rclone config to your brwoser and sign in,
cntrl + c to close the ssh tunnel

then either use the sync scripts, or mount it like below
```
rclone mount dropbox:/orgs-ecgr-QuantitativeImagingandAILaboratory /data1/dnicho26/EMG_DATASET/data --daemon -vv

```


This github demonstrates a project for an AI for Biomedical Applications final project. The idea was to send signals to one leg based off the location and signals read from the other leg. This can be used to restore gait for inviduals who may have nerve damage or other forms for physical disablities to correct their walk. This was preformed by taking 1.5 hours of walking data on two health indivudals using an EMG device which recorded eight muscle groups of each main muscle group as well as the acceleration of the sensor.

![Main live demonstration](/figures/demonstration.gif)

This walking data is all normalized and processed together before being seperated back out into files. This is to allow the data to scale relative to one another. Data was taken at multiple different walking speeds, 2.0mph, 2.5mph, 3.1mph, 5.0mph, 0.0mph. Which after processing is clearly shown in the image below.

![Emg of single muscle](/figures/emg_normalized.png)

This relativity allowed the signal to act as an intensity between 0 - 1 which can be used to set the intensity of potentiometers to deliver the correct intensity of eletrcity for the TENS units to stimulate the leg at. 

![Showing of gait pattern](/figures/legs_signal.gif)

# Models Used

An LSTM model was selected for its capacity to understand temporal data across long-term movements, such as a gait pattern, and their fine movement understanding for balancing and such. This model was particularly useful in the prediction of EMG to EMG signal data where having access to a live stream of data can accurately inference to send correct signals. The model used had a feature size of 28, a hidden size of 128, five layers, and an output size corresponding to the number of future time steps to predict, ten in this case.

![LSTM Training](/figures/training_loss.png)
![LSTM Ground and Predicted](/figures/Pred_Ground.PNG)

To utilize the pose information of the data set, two additional networks were explored Graph Convolution Networks and 3D Convolution Networks. The GCN has the ability to better capture information and the structure of a human pose. This can provide additional improvements in the mapping of pose to EMG data. As a baseline, however, a 3D Convolution Network was generated to better understand the temporal understanding of the movement over time during a sequence of frames. Neither model was able to train, however, likely due to a lack of data and a lack of available time for implementation. Future research would pretain to use these more advanced models to understand the characteristics of pose to EMG data.

# Installation 

Clone the repsistory into {ROOT}

`conda env create -f environment.yml`
This will install all required prerequisite for running the main body of the program
Next install pytorch, torchvision with your correct version of cuda or cpu

# Usage

![Original Plan on Black Board](/figures/original_plan.jpg)

All data is currently stored in the repository except for the video data, however this and the csvs will be moved to a different downloadable location. 

First step is to process the data, this file can be found in the data_processing folder. Please go within the file and change where your location for where the data is located. Also run the extract_skeleton file within the same folder to extract the skeleton infomation into npy files.

Next enter the training folder to view the LSTM training file. This will be used to train your LSTM on this dataset to predict further data.

Finally to test the LSTM connect an ardunio to your computer and set the following arduino code to run on your board setup to control your tens units or any other device you might have. Then run the test_signal file, ensure you select the correct usb serial connection before running. 

![Board](/figures/Board.jpg)

# Additional Information
Project was developed on Python 3.10 with windows 10
```
