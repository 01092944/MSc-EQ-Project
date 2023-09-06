# Python Heatmap visuliser program
# Created by Christopher Harrison
# Written on 05/07/2023

import pandas as pd
import folium

# Set working directory
import os
os.chdir("C:/Users/white/OneDrive/Desktop/Seismology/EQData")

# Read the CSV file
eq_data = pd.read_csv("eqWales100.csv")

# Create a folium map object
eq_map = folium.Map(location=[eq_data['lat'].mean(), eq_data['lon'].mean()], zoom_start=8)

# Add earthquake markers to the map
for index, row in eq_data.iterrows():
    folium.CircleMarker(location=[row['lat'], row['lon']], radius=row['ML'] * 2, color='red', fill=True).add_to(eq_map)

# Save the map as an HTML file
eq_map.save("eq_map.html")

import geopandas as gpd
os.chdir("C:/Users/white/OneDrive/Desktop/Seismology/EQData")
# Read the CSV file
eq_data = gpd.read_file("eqWales100.csv")

# Create a folium map object
eq_map = folium.Map(location=[eq_data['lat'].mean(), eq_data['lon'].mean()], zoom_start=8)

# Add earthquake markers to the map
for index, row in eq_data.iterrows():
    folium.CircleMarker(location=[row['lat'], row['lon']], radius=row['ML'] * 2, color='red', fill=True).add_to(eq_map)

# Save the map as an HTML file
eq_map.save("eq_map2.html")
