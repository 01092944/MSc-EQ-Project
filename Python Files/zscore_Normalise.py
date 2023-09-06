# Z-score normaliser
# Created by Christopher Harrison
# Created on 11/08/2023

import os
import pandas as pd

# Set the working directory to the folder containing the CSV files
os.chdir("C:/Users/white/OneDrive/Desktop/Seismology/EQData/")

# Function for Z-score normalization
def z_score_normalize(data):
    mean_val = data.mean()
    std_val = data.std()
    normalized_data = (data - mean_val) / std_val
    return normalized_data

# List of CSV files to process
csv_files = [
    "eqSouthWales.csv"
]

# Columns to normalize in each CSV file
columns_to_normalize = ["ML", "depth"]  # Replace with actual column names

# Normalize and save each CSV file separately
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    
    # Filter out records with ML less than 1.0
    df = df[df['ML'] >= 1.0]
    
    for column in columns_to_normalize:
        df[column] = z_score_normalize(df[column])
    
    # Extract the file name without extension
    file_name = os.path.splitext(csv_file)[0]
    
    # Save the filtered and normalized DataFrame to a new CSV file
    normalized_csv_file = f"{file_name}_filtered_zscore_normalized.csv"
    df.to_csv(normalized_csv_file, index=False)
    print(f"Filtered and Z-score normalized data saved as {normalized_csv_file}")

import pandas as pd
import matplotlib.pyplot as plt

# Read the normalized data
data = pd.read_csv("eqSouthWales_filtered_zscore_normalized.csv")  # Replace with your CSV file

# Create scatter plot for normalized magnitude vs. normalized depth
plt.scatter(data['ML'], data['depth'])
plt.xlabel('Normalized Magnitude')
plt.ylabel('Normalized Depth')
plt.title('Scatter Plot of Normalized Magnitude vs. Normalized Depth')
plt.grid(True)
plt.show()

