# Feature Correlation program Ver 2
# Written by Christopher Harrison
# Created on 10/08/2023

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set the working directory to the folder containing the CSV files
os.chdir("C:/Users/white/OneDrive/Desktop/Seismology/EQData/Correlation/Cluster Analysis 1")

# Dictionary mapping CSV file names to descriptive names
csv_name_mapping = {
    "1miEQMineColleriesswales_filtered_zscore_normalized.csv": "Mines & Colleries",
    "1miEQMineEntryswales_filtered_zscore_normalized.csv": "Mine Entries(CA)",
    "1miEQfaultswales_filtered_zscore_normalized.csv": "Faults",
    "1miEQdistfaultswales_filtered_zscore_normalized.csv": "Disturb Faults",
    "1miEQbreaksswales_filtered_zscore_normalized.csv": "Disturb Breaks/Fissures",
    "1miEQLakesswales_filtered_zscore_normalized.csv": "Lakes",
    "1miEQRiverswales_filtered_zscore_normalized.csv": "Rivers"
}

# List of CSV files to process
csv_files = list(csv_name_mapping.keys())

# Create an empty DataFrame to store all datasets
all_data = pd.DataFrame()

# Iterate through each CSV file
for csv_file in csv_files:
    # Read the dataset
    data = pd.read_csv(csv_file)

    # Store the dataset in the all_data DataFrame
    all_data[csv_name_mapping[csv_file]] = data['ML']

# Calculate correlation matrix for 'ML'
ml_correlation_matrix = all_data.corr()

# Print the 'ML' correlation matrix
print("Correlation Matrix - ML:")
print(ml_correlation_matrix)
print("\n")  # Add a newline for separation

# Save the 'ML' correlation matrix to a text file
ml_correlation_matrix.to_csv("ml_correlation_matrix.txt", sep=",")

# Plot 'ML' correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(ml_correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('ML Correlation Heatmap')
plt.show()

# Calculate correlation matrix for 'depth'
depth_correlation_matrix = all_data.corr()

# Print the 'depth' correlation matrix
print("Correlation Matrix - depth:")
print(depth_correlation_matrix)
print("\n")  # Add a newline for separation

# Save the 'depth' correlation matrix to a text file
depth_correlation_matrix.to_csv("depth_correlation_matrix.txt", sep=",")

# Plot 'depth' correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(depth_correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Depth Correlation Heatmap')
plt.show()
