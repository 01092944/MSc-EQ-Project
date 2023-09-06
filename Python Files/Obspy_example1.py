# ObsPy Analysis Program #1
# Written by Christopher Harrison
# Created on 13/08/2023

import os
from obspy import read_events
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import LAND, COASTLINE
import datetime

# Path to the QuakeML file
quakeml_file = 'C:/Users/white/OneDrive/Desktop/Seismology/EQData/python_tests/seismic_events.xml'

# Read the QuakeML file
catalog = read_events(quakeml_file)

# Extract latitude and longitude from events
latitudes = []
longitudes = []

for event in catalog:
    origin = event.origins[0]
    latitudes.append(origin.latitude)
    longitudes.append(origin.longitude)

# Plot event locations
plt.figure(figsize=(10, 6))
plt.scatter(longitudes, latitudes, c='r', marker='o', s=10)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Seismic Event Locations')
plt.grid()
plt.show()

# Plot event locations on a map using Cartopy
plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(LAND)
ax.add_feature(COASTLINE)
ax.scatter(longitudes, latitudes, transform=ccrs.PlateCarree(),
           c='r', marker='o', s=20)
ax.set_title('Seismic Event Locations')
plt.show()

# Extract magnitudes from events
magnitudes = []

for event in catalog:
    if event.magnitudes:
        magnitudes.append(event.magnitudes[0].mag)

# Plot histogram of magnitudes
plt.figure(figsize=(8, 6))
plt.hist(magnitudes, bins=20, color='blue', alpha=0.7)
plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.title('Histogram of Seismic Event Magnitudes')
plt.yscale('log')  # Use logarithmic scale for y-axis
plt.grid()
plt.show()

for event in catalog:
    depths = [origin.depth / 1000.0 for origin in event.origins if origin.depth]
    event_depths.extend(depths)

# Plot Depth Analysis
plt.figure(figsize=(10, 6))
plt.hist(event_depths, bins=np.arange(0, max(event_depths) + 10, 10), color='orange', alpha=0.7)
plt.xlabel('Depth (km)')
plt.ylabel('Frequency')
plt.title('Depth Analysis')
plt.grid()
plt.show()





