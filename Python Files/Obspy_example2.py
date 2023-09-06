# ObsPy Analysis Program #2
# Written by Christopher Harrison
# Created on 13/08/2023

import os
from obspy import read_events
import matplotlib.pyplot as plt
from collections import Counter
import datetime
import numpy as np

# Path to the QuakeML file
quakeml_file = 'C:/Users/white/OneDrive/Desktop/Seismology/EQData/python_tests/seismic_events.xml'

# Read the QuakeML file
catalog = read_events(quakeml_file)

# Extract event years from origins
event_years = []

for event in catalog:
    origin = event.origins[0]
    event_years.append(origin.time.year)

# Count the number of events per year
event_counter = Counter(event_years)

# Plot temporal analysis
years = list(event_counter.keys())
event_counts = list(event_counter.values())

plt.figure(figsize=(10, 6))
plt.bar(years, event_counts, color='green', alpha=0.7)
plt.xlabel('Year')
plt.ylabel('Number of Events')
plt.title('Temporal Analysis of Seismic Events')
plt.grid()
plt.show()


# Generate random magnitude data for demonstration
magnitude_min = np.random.uniform(2.5, 5.0, size=len(years))
magnitude_max = magnitude_min + np.random.uniform(0.1, 1.0, size=len(years))

# Convert years to datetime objects for matplotlib
dates = [datetime.datetime(year, 1, 1) for year in years]

# Create the candlestick plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.vlines(dates, magnitude_min, magnitude_max, color='black')
ax.hlines(magnitude_min, [date - datetime.timedelta(days=50) for date in dates],
          [date + datetime.timedelta(days=50) for date in dates], color='black')
ax.hlines(magnitude_max, [date - datetime.timedelta(days=50) for date in dates],
          [date + datetime.timedelta(days=50) for date in dates], color='black')
ax.xaxis_date()
ax.xaxis.set_major_formatter(dates.DateFormatter('%Y'))
plt.xlabel('Year')
plt.ylabel('Magnitude')
plt.title('Candlestick Plot of Seismic Event Magnitudes')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
