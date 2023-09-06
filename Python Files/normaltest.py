# Feature Correlation program
# Written by Christopher Harrison
# Created on 10/08/2023

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set the working directory to the folder containing the CSV files
os.chdir("C:/Users/white/OneDrive/Desktop/Seismology/EQData/Correlation/Cluster Analysis 1")

# List of CSV files to process
csv_files = [
    "1miEQMineColleriesswales_filtered_zscore_normalized.csv",
    "1miEQMineEntryswales_filtered_zscore_normalized.csv",
    "1miEQfaultswales_filtered_zscore_normalized.csv",
    "1miEQdistfaultswales_filtered_zscore_normalized.csv",
    "1miEQbreaksswales_filtered_zscore_normalized.csv",
    "1miEQLakesswales_filtered_zscore_normalized.csv",
    "1miEQRiverswales_filtered_zscore_normalized.csv"
]
    
# Iterate through each CSV file
for csv_file in csv_files:
    # Read the dataset
    data = pd.read_csv(csv_file)

    # Calculate correlation matrix
    correlation_matrix = data[['ML', 'depth']].corr()
    # Print the correlation matrix
    print(f"Correlation Matrix - {os.path.basename(csv_file)}:")
    print(correlation_matrix)
    print("\n")  # Add a newline for separation

    # Plot correlation matrix using a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title(f'Correlation Heatmap - {os.path.basename(csv_file)}')
    plt.show()
