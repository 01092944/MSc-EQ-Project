# ObsPy Analysis Program #7c - Waveform Visualiser
# Written by Christopher Harrison
# Created on 13/08/2023

import os
from obspy import read, UTCDateTime
import matplotlib.pyplot as plt

# Path to the directory containing HHZ waveform files
directory_path = 'C:/Users/white/OneDrive/Desktop/Seismology/EQData/HHZFiles/2019/MCH1/HHZ.D/001'

# Get a list of all files in the directory
file_list = [f for f in os.listdir(directory_path) if f.endswith('.001')]

# Loop through the files and analyze each one
for file_name in file_list:
    file_path = os.path.join(directory_path, file_name)
    
    # Read the waveform data
    st = read(file_path)
    
    # Print basic information about the waveform data
    print("File:", file_name)
    print("Number of Traces:", len(st))
    
    # Get the P-wave arrival time (starttime) and S-wave arrival time (endtime)
    p_arrival_time = st[0].stats.starttime
    s_arrival_time = st[0].stats.endtime
    
    # Calculate the duration
    duration = st[0].stats.endtime - st[0].stats.starttime
    
    # Print P-wave and S-wave arrival times, and duration
    print("P-wave Arrival Time:", p_arrival_time)
    print("S-wave Arrival Time:", s_arrival_time)
    print("Duration:", duration)
    
    # Plot the waveform data
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    st.plot(type='normal', title='Waveform: ' + file_name)
    
    # Calculate and plot the spectrogram
    plt.subplot(3, 1, 2)
    st.spectrogram(log=True, cmap='viridis')
    
    # Plot amplitude vs. time
    plt.subplot(3, 1, 3)
    plt.plot(st[0].times(), st[0].data)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Amplitude vs. Time')
    
    plt.tight_layout()
    plt.show()
