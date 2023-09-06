# ObsPy Analysis Program #4
# Written by Christopher Harrison
# Created on 13/08/2023

import os
from obspy import read_events
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import seaborn as sns
import numpy as np

# Path to the QuakeML file
quakeml_file = 'C:/Users/white/OneDrive/Desktop/Seismology/EQData/python_tests/seismic_events.xml'

# Read the QuakeML file
catalog = read_events(quakeml_file)

# Extract event locations (latitude and longitude) from the catalog
event_locations = [(origin.latitude, origin.longitude) for event in catalog for origin in event.origins]

# Calculate distances between event pairs
distances = []

for i in range(len(event_locations)):
    for j in range(i + 1, len(event_locations)):
        dist = geodesic(event_locations[i], event_locations[j]).kilometers
        distances.append(dist)

# Plot Distance Analysis
plt.figure(figsize=(10, 6))
plt.hist(distances, bins=20, color='purple', alpha=0.7)
plt.xlabel('Distance (km)')
plt.ylabel('Frequency')
plt.title('Distance Analysis')
plt.grid()
plt.show()

# Calculate distances between event pairs
distances = []

for i in range(len(event_locations)):
    for j in range(i + 1, len(event_locations)):
        dist = geodesic(event_locations[i], event_locations[j]).kilometers
        distances.append(dist)

# Plot Distance Analysis as a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(range(len(distances)), distances, color='blue', alpha=0.7, s=10)
plt.xlabel('Event Pair Index')
plt.ylabel('Distance (km)')
plt.title('Distance Analysis - Scatter Plot')
plt.grid()
plt.show()

# Plot Distance Analysis as a violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(y=distances, color='cyan', inner='stick')
plt.ylabel('Distance (km)')
plt.title('Distance Analysis - Violin Plot')
plt.grid()
plt.show()

# Calculate pairwise distances and create a distance matrix
num_events = len(event_locations)
distance_matrix = np.zeros((num_events, num_events))

for i in range(num_events):
    for j in range(i + 1, num_events):
        dist = geodesic(event_locations[i], event_locations[j]).kilometers
        distance_matrix[i, j] = dist
        distance_matrix[j, i] = dist

# Plot Pairwise Distance Matrix as a heatmap
plt.figure(figsize=(10, 6))
plt.imshow(distance_matrix, cmap='viridis', origin='lower')
plt.colorbar(label='Distance (km)')
plt.xlabel('Event Index')
plt.ylabel('Event Index')
plt.title('Pairwise Distance Matrix Heatmap')
plt.grid(False)
plt.show()
