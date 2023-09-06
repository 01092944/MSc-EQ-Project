# Cluster Analysis Version 2 for Rainfall versus Seismic Events
# Written by Christopher Harrison
# Created 04/08/2023
# Program for Cluster Analysis of Average Rainfall against Seismic Events

# Clear any data from R Studio
rm(list = ls())

# Set the working directory
setwd("C:/Users/white/OneDrive/Desktop/Seismology/EQData/Correlation")

# Required libraries
library(dplyr)
library(ggplot2)
library(rpart)
library(rpart.plot)

# Read the dataset (Replace 'path_to_file' with the actual file path)
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

# Calculate average depth for each day
average_depth <- data %>%
  group_by(EQDate) %>%
  summarize(Avg_Depth = mean(depth))

# Merge the data into a single dataframe
combined_data <- merge(average_ml, average_rainfall, by = "EQDate", all = TRUE)
combined_data <- merge(combined_data, average_depth, by = "EQDate", all = TRUE)

# Data Cleaning: Remove rows with missing values
combined_data <- combined_data[complete.cases(combined_data), ]

# Feature Engineering: Calculate relative earthquake magnitude
combined_data$Relative_ML <- combined_data$Avg_ML / max(combined_data$Avg_ML)

# Define the target variable (earthquake risk categories)
combined_data$Risk_Category <- cut(combined_data$Relative_ML,
                                   breaks = c(0, 0.33, 0.67, 1),
                                   labels = c("Low", "Medium", "High"),
                                   include.lowest = TRUE)

# Split the data into training and testing sets (80% training, 20% testing)
set.seed(123)  # For reproducibility
train_indices <- sample(1:nrow(combined_data), 0.8 * nrow(combined_data))
train_data <- combined_data[train_indices, ]
test_data <- combined_data[-train_indices, ]

# Build the Decision Tree Classification Model
model <- rpart(Risk_Category ~ Avg_Rainfall + Avg_Depth,
               data = train_data,
               method = "class")

# Evaluate the Model
predicted_risk <- predict(model, newdata = test_data, type = "class")

# Confusion matrix and accuracy
conf_matrix <- table(test_data$Risk_Category, predicted_risk)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)

# Visualize the Decision Tree
rpart.plot(model, box.palette = "limegreen")

# Print the confusion matrix and accuracy
print(conf_matrix)
##         predicted_risk
##          Low Medium High
##   Low      3      6    0
##   Medium   4      4    0
##   High     1      0    0
print(paste0("Accuracy: ", round(accuracy, 2) * 100, "%"))
