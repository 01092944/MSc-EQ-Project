# Mapview of Earthquake Data in South Wales over 100 Years
# Written by Christopher Harrison
# Created 05/07/2023


# Clear any data from R Studio
rm(list = ls())

# Set working directory
setwd("C:/Users/white/OneDrive/Desktop/Seismology/EQData")

# Read the CSV file
eq_data <- read.csv("eqWales100.csv", sep = ",")

# Define the longitude and latitude ranges for South Wales
south_wales_lon_min <- -5.4408
south_wales_lon_max <- -2.6394
south_wales_lat_min <- 51.3667
south_wales_lat_max <- 52.1908

# Filter data for South Wales coordinates
south_wales_data <- subset(eq_data, lon >= south_wales_lon_min & lon <= south_wales_lon_max &
                             lat >= south_wales_lat_min & lat <= south_wales_lat_max)

# Export filtered data to a new CSV file
write.csv(south_wales_data, file = "eqSouthWales.csv", row.names = FALSE)

# Load required libraries
library(mapview)
library(sf)

# Convert to spatial data frame
Magnitude <- st_as_sf(south_wales_data, coords = c("lon", "lat"), crs = 4326)

# Create a mapview object
eq_map <- mapview(Magnitude, zcol = "ML", map.types = "OpenStreetMap.DE")

# Display the map
eq_map
