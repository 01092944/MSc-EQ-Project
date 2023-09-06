# ObsPy Analysis Program #7b - Waveform Visualiser
# Written by Christopher Harrison
# Created on 13/08/2023

import os
from obspy import read, UTCDateTime
import matplotlib.pyplot as plt

# Path to the directory containing HHZ waveform files
directory_path = 'C:/Users/white/OneDrive/Desktop/Seismology/EQData/HHZFiles/2019/MCH1/HHZ.D/'

# Get a list of all files in the directory
file_list = [f for f in os.listdir(directory_path) if f.endswith('.320')]

# Loop through the files and analyze each one
for file_name in file_list:
    file_path = os.path.join(directory_path, file_name)
    
    # Read the waveform data
    st = read(file_path)
    
    # Print basic information about the waveform data
    print("File:", file_name)
    print("Number of Traces:", len(st))
    
    # Plot the waveform data
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    st.plot(type='normal', title='Waveform: ' + file_name)
    
    # Calculate and plot the spectrogram
    plt.subplot(2, 1, 2)
    st.spectrogram(log=True, cmap='viridis')
    
    # Show the plots
    plt.tight_layout()
    plt.show()
