# Factor Analysis Ver 1
# Written by Christopher Harrison
# Created by 06/08/2023

import pandas as pd
import os

# Set the directory containing factor datasets
factor_data_directory = "C:/Users/white/OneDrive/Desktop/Seismology/EQData/Correlation/Cluster Analysis 1"

# List of CSV files to process
csv_files = [
    "1miEQMineColleriesswales.csv",
    "1miEQMineEntryswales.csv",
    "1miEQfaultswales.csv",
    "1miEQdistfaultswales.csv",
    "1miEQbreaksswales.csv",
    "1miEQLakesswales.csv",
    "1miEQRiverswales.csv"
]

# Load earthquake data
earthquake_data = pd.read_csv("C:/Users/white/OneDrive/Desktop/Seismology/EQData/eqSouthWales_filtered_zscore_normalized.csv")


# Initialize an empty DataFrame to store correlations
correlations_df = pd.DataFrame(index=csv_files, columns=["ML", "depth"])

# Iterate through each factor dataset and calculate correlations
for csv_file in csv_files:
    factor_data = pd.read_csv(os.path.join(factor_data_directory, csv_file))

    # Convert non-numeric values in "intensity" column to NaN
    factor_data['intensity'] = pd.to_numeric(factor_data['intensity'], errors='coerce')
    
    # Drop rows with NaN values
    factor_data = factor_data.dropna(subset=['intensity'])
    
    # Calculate correlations
    correlations = earthquake_data[['ML', 'depth']].corrwith(factor_data)
    correlations_df.loc[csv_file] = correlations

# Print or analyze the correlations DataFrame as needed
print(correlations_df)
