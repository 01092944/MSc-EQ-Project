# Average rainfall analysis Version 1
# Written by Christopher Harrison
# Created 3/08/2023
# Program to analyse rainfall for correlations between rainfall and seismic activity.

# Clear any data from R Studio
rm(list = ls())

# Set the working directory
setwd("C:/Users/white/OneDrive/Desktop/Seismology/EQData/Correlation")

# Required libraries
library(dplyr)
library(lubridate)
library(ggplot2)
library(plotly)

# Read the dataset
path_to_file <- 'C:/Users/white/OneDrive/Desktop/Seismology/EQData/Correlation/rainfallEQcomplete.csv'
data <- read.csv(path_to_file)

# Convert EQDate column to proper date format
data$EQDate <- as.Date(data$EQDate)

# Calculate average ML for each day
average_ml <- data %>%
  group_by(EQDate) %>%
  summarize(Avg_ML = mean(ML))

# Calculate average rainfall for each day
average_rainfall <- data %>%
  group_by(EQDate) %>%
  summarize(Avg_Rainfall = mean(rainfall_mm))

# Merge average_ml and average_rainfall based on the date
combined_data <- merge(average_ml, average_rainfall, by = "EQDate", all = TRUE)

# Data Cleaning: Remove rows with missing values
combined_data <- combined_data[complete.cases(combined_data), ]

# Scatter plot of average rainfall against average ML
scatter_plot <- ggplot(combined_data, aes(x = Avg_Rainfall, y = Avg_ML)) +
  geom_point(color = 'steelblue', size = 3) +
  labs(title = "Average Rainfall vs. Average Earthquake Magnitude",
       x = "Average Rainfall",
       y = "Average Magnitude (ML)") +
  theme_minimal()

print(scatter_plot)
# Correlation between average rainfall and average ML
correlation_coefficient <- cor(combined_data$Avg_Rainfall, combined_data$Avg_ML)

# Draw a conclusion based on the correlation coefficient
if (correlation_coefficient > 0) {
  conclusion <- "There is a positive correlation between average rainfall and earthquake magnitude."
} else if (correlation_coefficient < 0) {
  conclusion <- "There is a negative correlation between average rainfall and earthquake magnitude."
} else {
  conclusion <- "There is no significant correlation between average rainfall and earthquake magnitude."
}

cat("\nConclusion:", conclusion, "\n")

# Linear regression analysis
linear_model <- lm(Avg_ML ~ Avg_Rainfall, data = combined_data)

# Print the summary of the linear regression model
print(summary(linear_model))
