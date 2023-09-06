# ObsPy Analysis Program #6
# Written by Christopher Harrison
# Created on 13/08/2023

import os
from obspy import read_events
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from sklearn.cluster import KMeans
os.environ["LOKY_MAX_CPU_COUNT"] = "14" # Computing deviced used to create the application has a max of 14 cores, adjust based on your own device
# Path to the QuakeML file
quakeml_file = 'C:/Users/white/OneDrive/Desktop/Seismology/EQData/python_tests/seismic_events.xml'

# Read the QuakeML file
catalog = read_events(quakeml_file)

# Extract event magnitudes and origin times from the catalog
event_magnitudes = []
event_times = []

for event in catalog:
    magnitude = event.magnitudes[0].mag if event.magnitudes else None
    if magnitude is not None:
        origin = event.origins[0]
        event_magnitudes.append(magnitude)
        event_times.append(origin.time.datetime)

# Plot Magnitude-Time Analysis
plt.figure(figsize=(10, 6))
plt.plot_date(date2num(event_times), event_magnitudes, markersize=5)
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.title('Magnitude-Time Analysis')
plt.grid()
plt.tight_layout()
plt.show()

# Extract event magnitudes
event_magnitudes = []

for event in catalog:
    magnitude = event.magnitudes[0].mag if event.magnitudes else None
    if magnitude is not None:
        event_magnitudes.append(magnitude)

# Plot Magnitude-Frequency Distribution
plt.figure(figsize=(10, 6))
plt.hist(event_magnitudes, bins=20, color='blue', alpha=0.7)
plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.title('Magnitude-Frequency Distribution')
plt.grid()
plt.tight_layout()
plt.show()


# Extract event locations (latitude and longitude)
event_locations = []

for event in catalog:
    origin = event.origins[0]
    event_locations.append((origin.latitude, origin.longitude))

# Use K-Means clustering to group events
num_clusters = 3  # Number of clusters to create
kmeans = KMeans(n_clusters=num_clusters)
clusters = kmeans.fit_predict(event_locations)

# Plot Event Clustering Map
plt.figure(figsize=(10, 6))
for cluster_idx in range(num_clusters):
    cluster_events = [event for i, event in enumerate(event_locations) if clusters[i] == cluster_idx]
    lats, lons = zip(*cluster_events)
    plt.scatter(lons, lats, label=f'Cluster {cluster_idx + 1}', alpha=0.7)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Event Clustering Map')
plt.legend()
plt.grid()
plt.show()
