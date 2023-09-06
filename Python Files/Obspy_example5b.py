# ObsPy Analysis Program #5b
# Written by Christopher Harrison
# Created on 13/08/2023

import os
from obspy import read_events
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Path to the QuakeML file
quakeml_file = 'C:/Users/white/OneDrive/Desktop/Seismology/EQData/python_tests/seismic_events.xml'

# Read the QuakeML file
catalog = read_events(quakeml_file)

# Define map boundaries (latitude and longitude)
lat_min = 51.0
lat_max = 52.5
lon_min = -4.0
lon_max = -2.5

# Create a Cartopy map
plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([lon_min, lon_max, lat_min, lat_max])

# Plot event density on the map
for event in catalog:
    origin = event.origins[0]
    plt.plot(origin.longitude, origin.latitude, 'ro', markersize=3, alpha=0.5, transform=ccrs.Geodetic())

# Add coastlines, countries, borders, and gridlines
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.gridlines(draw_labels=True)

plt.title('Regional Comparison: Event Density Map (Cartopy)')
plt.show()

