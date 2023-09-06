# Average rainfall analysis Version 3
# Written by Christopher Harrison
# Created 3/08/2023
# Improved program to analyse rainfall for correlations between rainfall and seismic activity.

# Clear any data from R Studio
rm(list = ls())

# Required libraries
library(dplyr)
library(lubridate)
library(ggplot2)
library(plotly)
library(tidyr)
library(knitr)

# Load data for Rainfall
rainfalldata <- read.csv('C:/Users/white/OneDrive/Desktop/Seismology/EQData/Correlation/rainfallEQcomplete.csv')

# Filter the rainfall data with ML >= 1.0
rainfalldata <- rainfalldata %>%
  filter(ML >= 1.0)

# View the top 10 records in rainfalldata as a formatted table
top_10_rainfall_table <- kable(head(rainfalldata,10))
print(top_10_rainfall_table)

# Set the directory where the CSV files are located
csv_directory <- 'C:/Users/white/OneDrive/Desktop/Seismology/EQData/Correlation/weatherfiltered'

# List all the CSV files in the directory
csv_files <- list.files(path = csv_directory, pattern = '\\.csv$', full.names = TRUE)

# Create an empty dataframe to store the combined data
combined_df <- data.frame()

# Loop through each CSV file and read them into a temporary dataframe
for (csv_file in csv_files) {
  temp_df <- read.csv(csv_file, header = TRUE)
  
  # Filter the CSV data with ML >= 1.0
  temp_df <- temp_df %>%
    filter(ML >= 1.0)
  
  # Convert the "yyyy.mm.dd" column to proper date format
  temp_df$yyyy.mm.dd <- as.Date(temp_df$yyyy.mm.dd, format = "%d/%m/%Y")  # Adjust the format if needed
  # Convert the "hh.mm.ss.ss" column to proper time format
  temp_df$hh.mm.ss.ss <- as.POSIXct(temp_df$hh.mm.ss.ss, format = "%H:%M:%OS")  # Adjust the format if needed
  # Convert the "locality" & "county" column to character type
  temp_df$locality <- as.character(temp_df$locality)
  temp_df$county <- as.character(temp_df$county)
  # Replace missing values in the "depth" column with 0
  temp_df$depth <- replace_na(temp_df$depth, 0)
  # Remove the "geometry" column
  temp_df <- select(temp_df, -geometry)
  
  # Combine the temporary dataframe with the existing combined dataframe
  combined_df <- bind_rows(combined_df, temp_df)
}

# Create a new dataframe to store the date data
date_data <- combined_df %>%
  select(yyyy.mm.dd) %>%
  distinct()

# Calculate average rainfall for each day
average_rainfall <- rainfalldata %>%
  group_by(EQDate) %>%
  summarize(Avg_Rainfall = mean(rainfall_mm))

# Calculate average ML for each day
average_ml <- combined_df %>%
  group_by(yyyy.mm.dd) %>%
  summarize(Avg_ML = mean(ML))

# Calculate average depth for each day
average_depth <- combined_df %>%
  group_by(yyyy.mm.dd) %>%
  summarize(Avg_Depth = mean(depth))

# Convert the "EQDate" column in average_rainfall to date format
average_rainfall$EQDate <- as.Date(average_rainfall$EQDate, format = "%d/%m/%Y")

# Merge the average rainfall, average ML, and average depth dataframes with date_data
combined_data <- left_join(date_data, average_rainfall, by = c("yyyy.mm.dd" = "EQDate"))
combined_data <- left_join(combined_data, average_ml, by = "yyyy.mm.dd")
combined_data <- left_join(combined_data, average_depth, by = "yyyy.mm.dd")

# Remove missing values from combined_data
combined_data <- na.omit(combined_data)

# Scatter plot of average rainfall against average ML with linear regression line
scatter_plot <- ggplot(combined_data, aes(x = Avg_Rainfall, y = Avg_ML)) +
  geom_point(color = 'steelblue', size = 3) +
  geom_smooth(method = "lm", color = 'red', se = FALSE) +  # Add linear regression line
  labs(title = "Average Rainfall vs. Average Earthquake Magnitude",
       x = "Average Rainfall",
       y = "Average Magnitude (ML)") +
  theme_minimal()

print(scatter_plot)

# Scatter plot of average rainfall against average depth with linear regression line
scatter_plot <- ggplot(combined_data, aes(x = Avg_Rainfall, y = Avg_Depth)) +
  geom_point(color = 'purple', size = 3) +
  geom_smooth(method = "lm", color = 'limegreen', se = FALSE) +  # Add linear regression line
  labs(title = "Average Rainfall vs. Average Earthquake Depth",
       x = "Average Rainfall",
       y = "Average Magnitude (ML)") +
  theme_minimal()

print(scatter_plot)

# Scatter plot of average ML against average depth with linear regression line
scatter_plot <- ggplot(combined_data, aes(x = Avg_ML, y = Avg_Depth)) +
  geom_point(color = '#FF99FF', size = 3) +
  geom_smooth(method = "lm", color = '#0099cc', se = FALSE) +  # Add linear regression line
  labs(title = "Average Earthquake Magnitude vs. Average Earthquake Depth",
       x = "Average Rainfall",
       y = "Average Magnitude (ML)") +
  theme_minimal()

print(scatter_plot)

# Box plot of earthquake magnitude (ML) grouped by rainfall levels
boxplot_ml <- ggplot(combined_data, aes(x = cut(Avg_Rainfall, breaks = c(0, 10, 20, 30, Inf)), y = Avg_ML)) +
  geom_boxplot(fill = 'skyblue', color = 'steelblue') +
  labs(title = "Box Plot of Earthquake Magnitude Grouped by Rainfall Levels",
       x = "Rainfall Levels",
       y = "Average Magnitude (ML)") +
  theme_minimal()

print(boxplot_ml)

# Box plot of earthquake depth grouped by rainfall levels
boxplot_depth <- ggplot(combined_data, aes(x = cut(Avg_Rainfall, breaks = c(0, 10, 20, 30, Inf)), y = Avg_Depth)) +
  geom_boxplot(fill = 'lightgreen', color = 'darkgreen') +
  labs(title = "Box Plot of Earthquake Depth Grouped by Rainfall Levels",
       x = "Rainfall Levels",
       y = "Average Depth") +
  theme_minimal()

print(boxplot_depth)

# Correlation between average rainfall and average ML
correlation_coefficientRvML <- cor(combined_data$Avg_Rainfall, combined_data$Avg_ML)

# Draw a conclusion based on the correlation coefficient
if (correlation_coefficientRvML > 0) {
  conclusion <- "There is a positive correlation between average rainfall and average earthquake magnitude."
} else if (correlation_coefficientRvML < 0) {
  conclusion <- "There is a negative correlation between average rainfall and average earthquake magnitude."
} else {
  conclusion <- "There is no significant correlation between average rainfall and average earthquake magnitude."
}

cat("\nConclusion:", conclusion, "\n")

# Correlation between average rainfall and average ML
correlation_coefficientRvD <- cor(combined_data$Avg_Rainfall, combined_data$Avg_Depth)

# Draw a conclusion based on the correlation coefficient
if (correlation_coefficientRvD > 0) {
  conclusion <- "There is a positive correlation between average rainfall and average earthquake depth."
} else if (correlation_coefficientRvD < 0) {
  conclusion <- "There is a negative correlation between average rainfall and average earthquake depth."
} else {
  conclusion <- "There is no significant correlation between average rainfall and average earthquake depth."
}

cat("\nConclusion:", conclusion, "\n")

# Correlation between average rainfall and average ML
correlation_coefficientMLvD <- cor(combined_data$Avg_ML, combined_data$Avg_Depth)

# Draw a conclusion based on the correlation coefficient
if (correlation_coefficientMLvD > 0) {
  conclusion <- "There is a positive correlation between average earthquake magnitude and average earthquake depth."
} else if (correlation_coefficientMLvD < 0) {
  conclusion <- "There is a negative correlation between average earthquake magnitude and average earthquake depth"
} else {
  conclusion <- "There is no significant correlation between average earthquake magnitude and average earthquake depth"
}

cat("\nConclusion:", conclusion, "\n")

# Fit a linear regression model
model <- lm(Avg_ML ~ Avg_Rainfall + Avg_Depth, data = combined_data)

# Summary of the regression model
summary(model)

# Residual plot for the linear regression model
residual_plot <- ggplot(combined_data, aes(x = Avg_Rainfall, y = resid(model))) +
  geom_point(color = 'blue', size = 3) +
  geom_hline(yintercept = 0, color = 'red', linetype = 'dashed') +
  labs(title = "Residual Plot for Linear Regression",
       x = "Average Rainfall",
       y = "Residuals") +
  theme_minimal()

print(residual_plot)

# Partial regression plot for the linear regression model
partial_plot <- ggplot(combined_data, aes(x = Avg_Rainfall, y = residuals(model) + Avg_ML)) +
  geom_point(color = 'green', size = 3) +
  geom_smooth(method = "lm", se = FALSE, color = 'blue') +
  labs(title = "Partial Regression Plot for Linear Regression",
       x = "Average Rainfall",
       y = "Partial Residuals") +
  theme_minimal()

print(partial_plot)

# Fitting Quadratic Polynomial Regression Model
model2 <- lm(Avg_ML ~ poly(Avg_Rainfall, 2) + poly(Avg_Depth, 2), data = combined_data)

# Summary of the model
summary(model2)

# Scatter plot of average rainfall against average ML with fitted regression curve
scatter_plot <- ggplot(combined_data, aes(x = Avg_Rainfall, y = Avg_ML)) +
  geom_point(color = '#000066', size = 3) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), se = FALSE, color = "#ffff33") +
  labs(title = "Average Rainfall vs. Average Earthquake Magnitude",
       x = "Average Rainfall",
       y = "Average Magnitude (ML)") +
  theme_minimal()

print(scatter_plot)

# Fitting Quadratic Polynomial Regression Model
model3 <- lm(Avg_ML ~ poly(Avg_Rainfall, 2) + poly(Avg_Depth, 2) + I(Avg_Rainfall^2) + I(Avg_Depth^2), data = combined_data)

# Summary of the model
summary(model3)

# Fitting Quadratic Polynomial Regression Model
model4 <- lm(Avg_ML ~ poly(Avg_Rainfall, 2) - poly(Avg_Depth, 2) * I(Avg_Rainfall^2) -I(Avg_Depth^2), data = combined_data)

# Summary of the model
summary(model4)

# Create a categorical variable for ML levels
combined_data <- combined_data %>%
  mutate(ML_Category = case_when(
    Avg_ML >= 1 & Avg_ML < 2 ~ "Low",
    Avg_ML >= 2 & Avg_ML < 3 ~ "Medium",
    Avg_ML >= 3 & Avg_ML < 4 ~ "High",
    Avg_ML >= 4 ~ "Extreme",
    TRUE ~ NA_character_
  ))

# Fit a linear regression model with ML Category as a predictor
model_with_category <- lm(Avg_ML ~ Avg_Rainfall + Avg_Depth + ML_Category, data = combined_data)

# Summary of the regression model
summary(model_with_category)

# Fit a linear regression model with ML Category as a predictor
model_with_category2 <- lm(Avg_ML ~ Avg_Rainfall +  ML_Category, data = combined_data)

# Summary of the regression model
summary(model_with_category2)

# Fit a linear regression model with ML Category as a predictor
model_with_category3 <- lm(Avg_ML ~ ML_Category * Avg_Rainfall, data = combined_data)

# Summary of the regression model
summary(model_with_category3)

# Create a residual plot graph
residual_plot_with_category <- ggplot(combined_data, aes(x = Avg_ML, y = resid(model_with_category), color = ML_Category)) +
  geom_point(size = 3) +
  geom_hline(yintercept = 0, color = 'red', linetype = 'dashed') +
  labs(title = "Residual Plot with ML Category for Linear Regression",
       x = "Average Magnitude (ML)",
       y = "Residuals",
       color = "ML Category") +
  theme_minimal()

print(residual_plot_with_category)

# Create a graph for predicted vs observed
predicted_vs_observed <- ggplot(combined_data, aes(x = Avg_ML, y = predict(model_with_category))) +
  geom_point(color = 'blue', size = 3) +
  geom_abline(intercept = 0, slope = 1, color = 'red', linetype = 'dashed') +
  labs(title = "Predicted vs. Observed Plot",
       x = "Observed Average Magnitude (ML)",
       y = "Predicted Average Magnitude") +
  theme_minimal()

print(predicted_vs_observed)

# Create a graph of residual vs fitted
residual_vs_fitted <- ggplot(combined_data, aes(x = predict(model_with_category), y = resid(model_with_category))) +
  geom_point(color = 'blue', size = 3) +
  geom_hline(yintercept = 0, color = 'red', linetype = 'dashed') +
  labs(title = "Residual vs. Fitted Plot",
       x = "Fitted Values",
       y = "Residuals") +
  theme_minimal()

print(residual_vs_fitted)

leverage_residual_plot <- ggplot(data.frame(leverage = hatvalues(model_with_category), standardized_residuals = rstudent(model_with_category)),
                                 aes(x = leverage, y = standardized_residuals)) +
  geom_point(color = 'blue', size = 3) +
  geom_smooth(method = 'lm', se = FALSE, color = 'red') +
  labs(title = "Leverage-Residual Plot",
       x = "Leverage",
       y = "Standardized Residuals") +
  theme_minimal()

print(leverage_residual_plot)


partial_regression_plot <- ggplot(combined_data, aes(x = Avg_Rainfall, y = Avg_ML - predict(model_with_category))) +
  geom_point(color = 'green', size = 3) +
  geom_smooth(method = "lm", se = FALSE, color = 'blue') +
  labs(title = "Partial Regression Plot",
       x = "Average Rainfall",
       y = "Partial Residuals") +
  theme_minimal()

print(partial_regression_plot)



