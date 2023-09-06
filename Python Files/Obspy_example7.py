# ObsPy Analysis Program #7 - Waveform Visualiser
# Written by Christopher Harrison
# Created on 13/08/2023

from obspy import read, UTCDateTime
import matplotlib.pyplot as plt

# Path to the HHZ waveform file
file_path = 'C:/Users/white/OneDrive/Desktop/Seismology/EQData/HHZFiles/2019/MCH1/HHZ.D/001/GB.MCH1.00.HHZ.D.2019.001.HHZ'

# Read the waveform data
st = read(file_path)

# Print basic information about the waveform data
print("File:", file_path)
print("Number of Traces:", len(st))

# Plot the waveform data
st.plot(type='normal', title=file_path)

# Calculate and plot the spectrogram
st.spectrogram(log=True, cmap='viridis')

# Show the plots
plt.show()
