WeatherDataMergerCleanUp.R
# Large Weather data set creator
# Written by Christopher Harrison
# Created 24/07/2023
# Program to merge all weather data frame created with DailyRainDataCleaner.r

# Clear any data from R Studio
rm(list = ls())

# Set working directory to the folder containing the CSV files
setwd("C:/Users/white/OneDrive/Desktop/Seismology/Weather Data/Daily Rain/Cleaned/MergedDF")

# Load required libraries
library(dplyr)

# Get a list of all CSV files in the working directory
file_list2 <- list.files(pattern = "*.csv$")

# Read and merge the CSV files
merged_data2 <- lapply(file_list2, read.csv, header = TRUE) %>%
  bind_rows()

# Sort the merged data by 'lat' and 'lon'
merged_data2 <- merged_data2 %>%
  arrange(lat, lon)

# Drop specified columns
columns_to_drop <- c("ob_day_cnt", "src_id", "rec_st_ind", "ob_day_cnt_q", "prcp_amt_q", "prcp_amt_j", "meto_stmp_time", "midas_stmp_etime")
merged_data2 <- merged_data2 %>%
  select(-one_of(columns_to_drop))

# Set working directory to "Output" folder
setwd("Output")

# Save the sorted and merged data to a new CSV file named "Weather_Final.csv"
write.csv(merged_data2, file = "Weather_Final2.csv", row.names = FALSE)

# Print a message indicating successful completion
cat("Merging, sorting, dropping columns, and saving to Weather_Final.csv is complete!\n")
