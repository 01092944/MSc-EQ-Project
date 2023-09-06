# Program to create a spatial index
# Written by Christopher Harrison
# Created 10/08/2023

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Path to the CSV file
csv_file_path = 'C:/Users/white/OneDrive/Desktop/Seismology/EQData/eqSouthWales.csv'

# Specify the column names in the CSV file
column_names = ['yyyy.mm.dd', 'hh.mm.ss.ss', 'lat', 'lon', 'depth', 'ML', 'Nsta', 'RMS', 'intensity', 'induced', 'locality', 'county']

# Read the CSV file into a pandas DataFrame
data = pd.read_csv(csv_file_path, delimiter='\t', parse_dates={'datetime': ['yyyy.mm.dd', 'hh.mm.ss.ss']}, names=column_names, dayfirst=True, skiprows=1)

# Convert latitude and longitude to Point geometries
geometry = [Point(xy) for xy in zip(data['lon'], data['lat'])]

# Create a GeoDataFrame with the data and geometry
gdf = gpd.GeoDataFrame(data, geometry=geometry, crs='EPSG:4326')

# Save the GeoDataFrame to a new CSV file
new_csv_file_path = 'C:/Users/white/OneDrive/Desktop/Seismology/EQData/eqSouthWales_v2.csv'
gdf.to_csv(new_csv_file_path, index=False)

print("GeoDataFrame saved to:", new_csv_file_path)
