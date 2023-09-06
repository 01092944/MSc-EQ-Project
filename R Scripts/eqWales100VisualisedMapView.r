# Mapview of Earthquake Data in Wales over 100 Years
# Written by Christopher Harrison
# Created 05/07/2023

# Clear any data from R Studio
rm(list = ls())

# Load required libraries
library(mapview)
library(sf)

# Set working directory - Change to suit location of data used.
setwd("C:/Users/white/OneDrive/Desktop/Seismology/EQData")

# Read the CSV file
eq_data <- read.csv("eqWales100.csv", sep = ",")

# Convert to spatial data frame
Magnitude <- st_as_sf(eq_data, coords = c("lon", "lat"), crs = 4326)

# Create a mapview object
eq_map <- mapview(Magnitude, zcol = "ML", map.types = "Esri.WorldImagery")

# Display the map
eq_map
