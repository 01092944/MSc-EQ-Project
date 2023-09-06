# Cluster Analysis Version 1 for Rainfall versus Seismic Events
# Written by Christopher Harrison
# Created 04/08/2023
# program for Cluster Analysis of Average Rainfall against Seismic Events


# Clear any data from R Studio
rm(list = ls())

# Set the working directory
setwd("C:/Users/white/OneDrive/Desktop/Seismology/EQData/Correlation")

# Required libraries
library(dplyr)
library(ggplot2)
library(plotly)
library(cluster)

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

# Select the columns for clustering
clustering_data <- combined_data[, c("Avg_ML", "Avg_Rainfall", "Avg_Depth")]

# Perform K-means clustering with k=3 (you can change k to any desired number of clusters)
k <- 3
set.seed(123)  # For reproducibility
cluster_model <- kmeans(clustering_data, centers = k)

# Add the cluster assignments to the combined_data dataframe
combined_data$Cluster <- as.factor(cluster_model$cluster)

# Visualize the clusters
scatter_plot <- ggplot(combined_data, aes(x = Avg_Rainfall, y = Avg_ML, color = Cluster)) +
  geom_point(size = 3) +
  labs(title = "K-means Clustering of Average Rainfall vs. Average Earthquake Magnitude",
       x = "Average Rainfall",
       y = "Average Magnitude (ML)",
       color = "Cluster") +
  theme_minimal()

print(scatter_plot)

# You can also explore the characteristics of each cluster by examining the cluster centroids:
cluster_centroids <- as.data.frame(cluster_model$centers)
print(cluster_centroids)
##     Avg_ML Avg_Rainfall Avg_Depth
## 1 1.772222    139.01111  6.100000
## 2 1.340625     56.33250  6.549583
## 3 1.282927     14.41463  8.503252
# Cluster Profiles Plot
cluster_profiles_plot <- ggplot(combined_data, aes(x = Cluster, y = Avg_ML, color = Cluster)) +
  geom_boxplot() +
  labs(title = "Cluster Profiles of Average Earthquake Magnitude",
       x = "Cluster",
       y = "Average Magnitude (ML)",
       color = "Cluster") +
  theme_minimal()

print(cluster_profiles_plot)

# Hierarchical Clustering (Dendrogram)
hierarchical_clusters <- hclust(dist(clustering_data), method = "ward.D2")
dendrogram_plot <- as.dendrogram(hierarchical_clusters)
plot(dendrogram_plot, main = "Dendrogram of Clusters", xlab = "Sample", ylab = "Distance")

# Parallel Coordinate Plot
library(GGally)
## Registered S3 method overwritten by 'GGally':
##   method from   
##   +.gg   ggplot2
parallel_plot <- ggparcoord(combined_data, columns = 2:4, groupColumn = "Cluster")
print(parallel_plot)

