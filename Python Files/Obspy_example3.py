# ObsPy Analysis Program #3
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

# Extract event depths from the catalog
event_depths = []

for event in catalog:
    depths = [origin.depth / 1000.0 for origin in event.origins if origin.depth is not None and origin.depth != "N/A"]
    event_depths.extend(depths)

if not event_depths:
    print("No valid depth data found in the catalog.")
else:
    # Plot Depth Analysis
    plt.figure(figsize=(10, 6))
    plt.hist(event_depths, bins=np.arange(0, max(event_depths) + 10, 10), color='orange', alpha=0.7)
    plt.xlabel('Depth (km)')
    plt.ylabel('Frequency')
    plt.title('Depth Analysis')
    plt.grid()
    plt.show()
