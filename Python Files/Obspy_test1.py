# ObsPy Analysis Program #1
# Written by Christopher Harrison
# Created on 13/08/2023
# Program will connect to seismic station

import obspy
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import matplotlib.pyplot as plt

# Specify the FDSN web service URL
client = Client("IRIS")

# Define the start and end times for the data
start_time = UTCDateTime(2023, 2, 1)  # Replace with your desired start date
end_time = UTCDateTime(2023, 3, 31)    # Replace with your desired end date

# Specify the network, station, location, and channel codes
network = "GB"
station = "MCH1"
location = ""   # No specific location code for this station, change if available
channel = "HHZ"  # The channel code you want to access

# Fetch the waveform data
waveform = client.get_waveforms(network, station, location, channel, start_time, end_time)

# Plot the waveform
plt.figure(figsize=(10, 6))
waveform.plot()
plt.title(f"{network}.{station}.{channel}")
plt.show()
