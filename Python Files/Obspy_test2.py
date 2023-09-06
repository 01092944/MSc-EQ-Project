# ObsPy Analysis Program 
# Written by Christopher Harrison
# Created on 13/08/2023
# Program used to create .xml file for further analysis.

import os
import pandas as pd
from obspy import UTCDateTime
from obspy.core.event import Event, Origin, Magnitude

# Path to the CSV file
csv_file_path = 'C:/Users/white/OneDrive/Desktop/Seismology/EQData/eqSouthWalesfiltered.csv'

# Read CSV data using pandas
df = pd.read_csv(csv_file_path, delimiter=',')

# Convert CSV data to ObsPy Event objects
events = []

for index, row in df.iterrows():
    event = Event()

    # Reformat the date-time string to a format that UTCDateTime understands
    date_components = row['yyyy.mm.dd'].split('/')
    formatted_date = f"{date_components[2]}-{date_components[1]}-{date_components[0]}"
    
    origin_time = UTCDateTime(formatted_date + ' ' + row['hh.mm.ss.ss'])
    latitude = float(row['lat'])
    longitude = float(row['lon'])
    magnitude = float(row['ML'])
    
    origin = Origin()
    origin.time = origin_time
    origin.latitude = latitude
    origin.longitude = longitude
    
    mag = Magnitude()
    mag.mag = magnitude
    
    event.origins.append(origin)
    event.magnitudes.append(mag)
    
    events.append(event)

# Save ObsPy Event objects to QuakeML file
output_filename = 'seismic_events.xml'
output_path = os.path.join(os.getcwd(), output_filename)

from obspy import Catalog
cat = Catalog(events=events)
cat.write(output_path, format='QUAKEML')

print(f"Processed {len(events)} events. QuakeML file saved at: {output_path}")
