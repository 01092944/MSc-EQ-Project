# Average rainfall analysis Version 1
# Written by Christopher Harrison
# Created 3/08/2023
# Program to analyse rainfall for correlations between rainfall and seismic activity.

# Average rainfall analysis Version 2
# Written by Christopher Harrison
# Created 3/08/2023
# Modified program to analyse rainfall for correlations between rainfall and seismic activity.

# Clear any data from R Studio
rm(list = ls())

# Set the working directory
setwd("C:/Users/white/OneDrive/Desktop/Seismology/EQData/Correlation")

# Required libraries
library(dplyr)
library(lubridate)
library(ggplot2)
library(plotly)

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

average_depth <- data %>%
  group_by(EQDate) %>%
  summarize(Avg_Depth = mean(depth))

# Merge 'average_ml' and 'average_rainfall' based on the 'EQDate'
combined_data <- merge(average_ml, average_rainfall, by = "EQDate", all = TRUE)

# Merge 'combined_data' with 'average_depth' based on the 'EQDate'
combined_data <- merge(combined_data, average_depth, by = "EQDate", all = TRUE)

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

# Regression analysis with feature selection and interaction terms
model <- lm(Avg_ML ~ Avg_Rainfall + I(Avg_Rainfall^2), data = combined_data)
summary(model)

# Check for outliers and their influence
outliers <- cooks.distance(model) > 4/length(model$coefficients)
influential_outliers <- subset(combined_data, outliers)
print(influential_outliers)


# Interaction term between Avg_Rainfall and other features (Replace 'column2' and 'column3' with actual column names)
model_with_depth <- lm(Avg_ML ~ Avg_Rainfall + I(Avg_Rainfall^2) / I(Avg_Depth^2), data = combined_data)
summary(model_with_depth)

# Feature selection using stepwise regression
stepwise_model <- step(model_with_depth)
# Print the summary of the final model
summary(stepwise_model)

# Fit the refined model
refined_model <- lm(Avg_ML ~ Avg_Rainfall + I(Avg_Rainfall^2) + Avg_Depth + I(Avg_Depth^2) + Avg_Rainfall:Avg_Depth, data = combined_data)

# Print the summary of the refined model
summary(refined_model)

# Feature selection using stepwise regression
stepwise_model2 <- step(refined_model)

# Print the summary of the final model
summary(stepwise_model2)

library(randomForest)

# Build the Random Forest Regression model
model_rf <- randomForest(Avg_ML ~ Avg_Rainfall + Avg_Depth + I(Avg_Rainfall^2) + I(Avg_Depth^2) + Avg_Rainfall:Avg_Depth, data = combined_data)

# Print the summary of the model
print(model_rf)

# Make predictions using the model
predicted_ml <- predict(model_rf, combined_data)

# Calculate the R-squared value
rsquared <- cor(combined_data$Avg_ML, predicted_ml)^2
cat("R-squared value:", rsquared, "\n")

library(gbm)

# Build the Gradient Boosting Regression model
model_gbm <- gbm(Avg_ML ~ Avg_Rainfall + Avg_Depth + I(Avg_Rainfall^2) + I(Avg_Depth^2) + Avg_Rainfall:Avg_Depth,
                 data = combined_data,
                 n.trees = 500,
                 interaction.depth = 4,
                 shrinkage = 0.01,
                 verbose = TRUE)

# Summary of the model
summary(model_gbm)

# Load required libraries
library(e1071)

# Create the SVR model
svr_model <- svm(Avg_ML ~ Avg_Rainfall + Avg_Depth + I(Avg_Rainfall^2) + I(Avg_Depth^2) + Avg_Rainfall:Avg_Depth,
                 data = combined_data)

# Print the summary of the SVR model
summary(svr_model)

# Set the seed for reproducibility
set.seed(42)

# Create indices for train and test split (80% train, 20% test)
train_indices <- sample(1:nrow(combined_data), 0.8 * nrow(combined_data))
train_data <- combined_data[train_indices, ]
test_data <- combined_data[-train_indices, ]

# Train the SVR model
svr_model <- svm(Avg_ML ~ Avg_Rainfall + Avg_Depth + I(Avg_Rainfall^2) + I(Avg_Depth^2) + Avg_Rainfall:Avg_Depth,
                 data = train_data,
                 type = "eps-regression",
                 kernel = "radial",
                 cost = 1,
                 gamma = 0.2,
                 epsilon = 0.1)

# Make predictions on the test set
predictions <- predict(svr_model, test_data)

# Evaluate the model
mse <- mean((test_data$Avg_ML - predictions)^2)
rmse <- sqrt(mse)
mae <- mean(abs(test_data$Avg_ML - predictions))
r_squared <- cor(test_data$Avg_ML, predictions)^2

# Print evaluation metrics
cat("Mean Squared Error (MSE):", mse, "\n")
cat("Root Mean Squared Error (RMSE):", rmse, "\n")
cat("Mean Absolute Error (MAE):", mae, "\n")
cat("R-squared:", r_squared, "\n")

# Split the data into training and testing sets
set.seed(123) # For reproducibility
train_indices <- sample(1:nrow(combined_data), 0.7 * nrow(combined_data))
train_data <- combined_data[train_indices, ]
test_data <- combined_data[-train_indices, ]

# Build the Random Forest regression model
rf_model <- randomForest(Avg_ML ~ Avg_Rainfall + Avg_Depth + I(Avg_Rainfall^2) + I(Avg_Depth^2) + Avg_Rainfall:Avg_Depth,
                         data = train_data, ntree = 500)

# Make predictions on the test data
predictions <- predict(rf_model, test_data)

# Evaluate the model
mse <- mean((predictions - test_data$Avg_ML)^2)
rmse <- sqrt(mse)
mae <- mean(abs(predictions - test_data$Avg_ML))
r_squared <- cor(predictions, test_data$Avg_ML)^2

# Print the evaluation metrics
cat("Mean Squared Error (MSE):", mse, "\n")
cat("Root Mean Squared Error (RMSE):", rmse, "\n")
cat("Mean Absolute Error (MAE):", mae, "\n")
cat("R-squared:", r_squared, "\n")

set.seed(123) # For reproducibility
train_indices <- sample(1:nrow(combined_data), 0.8 * nrow(combined_data))
train_data <- combined_data[train_indices, ]
test_data <- combined_data[-train_indices, ]

# Fit the Gradient Boosting model
model_gbm <- gbm(Avg_ML ~ Avg_Rainfall + Avg_Depth + I(Avg_Rainfall^2) + I(Avg_Depth^2) + Avg_Rainfall:Avg_Depth, 
                 data = train_data, 
                 n.trees = 500,
                 interaction.depth = 4,
                 shrinkage = 0.01,
                 verbose = FALSE)

# Make predictions on the test data
predictions_gbm <- predict(model_gbm, newdata = test_data, n.trees = 500)

# Calculate evaluation metrics
mse_gbm <- mean((predictions_gbm - test_data$Avg_ML)^2)
rmse_gbm <- sqrt(mse_gbm)
mae_gbm <- mean(abs(predictions_gbm - test_data$Avg_ML))
r_squared_gbm <- cor(predictions_gbm, test_data$Avg_ML)^2

# Print the evaluation metrics
cat("Mean Squared Error (MSE) - Gradient Boosting:", mse_gbm, "\n")
cat("Root Mean Squared Error (RMSE) - Gradient Boosting:", rmse_gbm, "\n")
cat("Mean Absolute Error (MAE) - Gradient Boosting:", mae_gbm, "\n")
cat("R-squared - Gradient Boosting:", r_squared_gbm, "\n")


# Train-test split
set.seed(42) # For reproducibility
train_indices <- sample(1:nrow(combined_data), 0.8 * nrow(combined_data))
train_data <- combined_data[train_indices, ]
test_data <- combined_data[-train_indices, ]

# Train the Random Forest model
rf_model <- randomForest(Avg_ML ~ Avg_Rainfall + Avg_Depth + I(Avg_Rainfall^2) + I(Avg_Depth^2) + Avg_Rainfall:Avg_Depth, data = train_data, ntree = 500)

# Make predictions on the test data
rf_predictions <- predict(rf_model, newdata = test_data)

# Evaluate the model
mse_rf <- mean((rf_predictions - test_data$Avg_ML)^2)
rmse_rf <- sqrt(mse_rf)
mae_rf <- mean(abs(rf_predictions - test_data$Avg_ML))
r_squared_rf <- 1 - (mse_rf / var(test_data$Avg_ML))

# Print the evaluation metrics
cat("Mean Squared Error (MSE) - Random Forest:", mse_rf, "\n")
cat("Root Mean Squared Error (RMSE) - Random Forest:", rmse_rf, "\n")
cat("Mean Absolute Error (MAE) - Random Forest:", mae_rf, "\n")
cat("R-squared - Random Forest:", r_squared_rf, "\n")




