# Factor Analysis Ver 2
# Written by Christopher Harrison
# Created by 06/08/2023

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the directory containing factor datasets
factor_data_directory = "C:/Users/white/OneDrive/Desktop/Seismology/EQData/Correlation/Cluster Analysis 1"

# List of CSV files to process
csv_files = [
    "1miEQMineColleriesswales.csv",
    "1miEQMineEntryswales.csv",
    "1miEQfaultswales.csv",
    "1miEQdistfaultswales.csv",
    "1miEQbreaksswales.csv",
    #"1miEQLakesswales.csv",
    "1miEQRiverswales.csv"
]

# Load earthquake data
earthquake_data = pd.read_csv("C:/Users/white/OneDrive/Desktop/Seismology/EQData/eqSouthWales_filtered_zscore_normalized.csv")

# Initialize an empty DataFrame to store correlations
correlations_df = pd.DataFrame(index=csv_files, columns=["ML", '''"depth"'''])

# ...

# Initialize dictionaries to store correlation metrics
correlation_metrics = {
    "Pearson's r": [],
    "Spearman's rank": [],
    "Kendall's Tau": []
}

# Iterate through each factor dataset and calculate correlations
for csv_file in csv_files:
    factor_data = pd.read_csv(os.path.join(factor_data_directory, csv_file))
    
    # Convert non-numeric values in "intensity" column to NaN
    factor_data['intensity'] = pd.to_numeric(factor_data['intensity'], errors='coerce')
    
    # Calculate correlations only if "intensity" column has sufficient data
    if factor_data['intensity'].notna().sum() > 0:
        # Drop rows with missing "intensity" data
        factor_data = factor_data.dropna(subset=['intensity'])
        
        # Exclude constant columns from factor_data
        factor_data = factor_data.loc[:, factor_data.nunique() > 1]
        
        # Check if any remaining columns are left for correlation analysis
        if factor_data.shape[1] > 1:
            # Calculate correlations
            correlations = earthquake_data[['ML', '''depth''']].corrwith(factor_data)
            correlations_df.loc[csv_file] = correlations
            
            # Calculate different correlation metrics
            pearson_corr = correlations['ML']
            spearman_corr = factor_data.corrwith(earthquake_data['ML'], method='spearman')['intensity']
            kendall_corr = factor_data.corrwith(earthquake_data['ML'], method='kendall')['intensity']
            
            correlation_metrics["Pearson's r"].append(pearson_corr)
            correlation_metrics["Spearman's rank"].append(spearman_corr)
            correlation_metrics["Kendall's Tau"].append(kendall_corr)
        else:
            print(f"Warning: No valid columns left for correlation analysis in {csv_file}")
    else:
        print(f"Warning: Insufficient 'intensity' data for correlation analysis in {csv_file}")

# Print correlations DataFrame
print(correlations_df)

# Print correlation metrics
for metric, values in correlation_metrics.items():
    print(f"{metric}: {np.mean(values)}")

# Scatter plot of earthquake magnitude and intensity
plt.figure(figsize=(8, 6))
sns.scatterplot(x=earthquake_data['ML'], y=earthquake_data['intensity'])
plt.xlabel('Earthquake Magnitude (ML)')
plt.ylabel('Intensity')
plt.title('Scatter Plot of Earthquake Magnitude vs Intensity')
plt.grid(True)
plt.show()
