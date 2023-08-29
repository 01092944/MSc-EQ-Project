# Time-lapse video creation program
# Written by Christopher Harrison
# Created on 16/07/2023

# Clear any data from R Studio
rm(list = ls())

library(ggplot2)
library(maps)
library(lubridate)

# Set the longitude and latitude ranges for South Wales
south_wales_lon_min <- -5.4408
south_wales_lon_max <- -2.6394
south_wales_lat_min <- 51.3667
south_wales_lat_max <- 52.1908

# Set working directory - change to reflect where you will load the files from
setwd("C:/Users/white/OneDrive/Desktop/Seismology/EQData")

# Read the sample dataset
data <- read.csv("tlap0030.csv", sep = ",", stringsAsFactors = FALSE) #change file name to match tlapxxxx.csv files before running application

# Create a map plot as the background
map_plot <- ggplot() +
  borders("world", xlim = c(south_wales_lon_min, south_wales_lon_max),
          ylim = c(south_wales_lat_min, south_wales_lat_max), fill = "lightgray") +
  coord_cartesian(xlim = c(south_wales_lon_min, south_wales_lon_max),
                  ylim = c(south_wales_lat_min, south_wales_lat_max))

# Create the 'png' directory if it doesn't exist
if (!dir.exists("png")) {
  dir.create("png")
}

# Create a time-lapse of individual records
for (i in 1:nrow(data)) {
  # Plot each record on top of the map
  record_plot <- map_plot +
    geom_point(data = data[i, ], aes(x = lon, y = lat, color = ML)) +
    labs(title = paste("Record:", i),
         subtitle = paste("Locality:", data[i, "locality"]),
         caption = paste("Date:", data[i, "yyyy.mm.dd"], "Time:", data[i, "hh.mm.ss.ss"]))
  
  # Save the plot as a PNG file in the 'png' directory
  ggsave(filename = paste0("png/record_", i, ".png"), plot = record_plot)
}

# Get the locality name, current date, and time
locality <- data[1, "locality"]
current_date <- format(Sys.Date(), "%Y%m%d")
current_time <- format(Sys.time(), "%H%M%S")

# Define the output MP4 file name based on locality, date, and time
output_mp4 <- paste0(locality, "_", current_date, "_", current_time, ".mp4")

# Execute the 'ffmpeg' command to merge PNG files into an MP4 video with specific codecs
command <- paste("ffmpeg -framerate 1 -i png/record_%d.png -c:v libx264 -pix_fmt yuv420p -crf 23 -preset medium -c:a aac -b:a 192k -y", output_mp4)
system(command)

# Delete the PNG files
file.remove(list.files("png", pattern = "\\.png$", full.names = TRUE))

