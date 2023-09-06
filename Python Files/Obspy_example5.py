# ObsPy Analysis Program #5
# Written by Christopher Harrison
# Created on 13/08/2023

import os
from obspy import read_events
import matplotlib.pyplot as plt
import numpy as np

# Path to the QuakeML file
quakeml_file = 'C:/Users/white/OneDrive/Desktop/Seismology/EQData/python_tests/seismic_events.xml'

# Read the QuakeML file
catalog = read_events(quakeml_file)

# Define boundaries of the rectangular grid (latitude and longitude)
min_lat = 51.0
max_lat = 52.5
min_lon = -3.5
max_lon = -2.0
grid_resolution = 0.1  # in degrees

# Create a grid
lats = np.arange(min_lat, max_lat, grid_resolution)
lons = np.arange(min_lon, max_lon, grid_resolution)
event_density = np.zeros((len(lats), len(lons)))

# Calculate event density within the grid cells
for event in catalog:
    origin = event.origins[0]
    lat_idx = int((origin.latitude - min_lat) / grid_resolution)
    lon_idx = int((origin.longitude - min_lon) / grid_resolution)
    if 0 <= lat_idx < len(lats) and 0 <= lon_idx < len(lons):
        event_density[lat_idx, lon_idx] += 1

# Plot Regional Comparison - Event Density Map
plt.figure(figsize=(10, 6))
plt.imshow(event_density, cmap='plasma', extent=[min_lon, max_lon, min_lat, max_lat], origin='lower', aspect='auto')
plt.colorbar(label='Event Density')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Regional Comparison - Event Density Map')
plt.grid(False)
plt.show()
