# Weather Data file creator
# Written by Christopher Harrison
# Created 21/07/2023
# Program to merge multiple weather station files into one data frame, while removing unwanted data.
# End file needs to be manually cleaned to remove blank and duplicate header rows.


# Clear any data from R Studio
rm(list = ls())

# Set the working directory. Change to meet the correct directory being worked with
setwd("C:/Users/white/OneDrive/Desktop/Seismology/Weather Data/Daily Rain/south-glamorgan/19206_st-athan/qc-version-0")

# List all CSV files in the directory
csv_files <- list.files(pattern = "*.csv")

# Create an empty data frame to store the cleaned data
cleaned_data <- data.frame()

# Loop through each CSV file
for (file in csv_files) {
  # Read the CSV file and skip the rows with column names
  data <- read.csv(file, header = FALSE, sep = ",", skip = 61)
  
  # Drop unwanted columns
  unwanted_columns <- c("Conventions", "G", "BADC-CSV", "1",
                        "met_domain_name", "ob_end_ctime", "rec_st_ind",
                        "prcp_amt_q", "prcp_amt_j", "meto_stmp_time", "midas_stmp_etime")
  data <- data[, !(names(data) %in% unwanted_columns)]
  
  # Rename columns
  colnames(data) <- c("date", "id", "id_type", "version_num", "met_domain_name",
                      "ob_end_ctime", "ob_day_cnt", "src_id", "rec_st_ind",
                      "prcp_amt", "ob_day_cnt_q", "prcp_amt_q", "prcp_amt_j",
                      "meto_stmp_time", "midas_stmp_etime")
  
  # Add new columns. Update to match the correct information for the data sets you plan to merge.
  location <- "19206_st-athan"
  observation_station <- "st-athan"
  historic_county_name <- "south-glamorgan"
  height <- 49
  
  data$lat <- rep(51.405, nrow(data))
  data$lon <- rep(-3.441, nrow(data))
  data$stat_name <- rep(observation_station, nrow(data))
  data$hist_county_name <- rep(historic_county_name, nrow(data))
  data$height <- rep(height, nrow(data))
  
  # Append the cleaned data to the main data frame
  cleaned_data <- rbind(cleaned_data, data)
}

# Set the output directory
output_directory <- "C:/Users/white/OneDrive/Desktop/Seismology/Weather Data/Daily Rain/Cleaned"

# Write the cleaned data to a new CSV file
output_file <- paste0(output_directory, "/19206_st-athan.csv")
write.csv(cleaned_data, file = output_file, row.names = FALSE)

# Print a message indicating the process is completed
cat("Data cleaning and merging completed. Cleaned data stored in:", output_file, "\n")
