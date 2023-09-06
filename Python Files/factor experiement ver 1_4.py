# Factor Analysis Ver 4
# Written by Christopher Harrison
# Created by 11/08/2023

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
import folium
from folium import plugins
import statsmodels.api as sm


# Set the directory containing factor datasets
factor_data_directory = "C:/Users/white/OneDrive/Desktop/Seismology/EQData/Correlation/Cluster Analysis 1/"

mine_colleries_path = factor_data_directory + "1miEQMineColleriesswales_filtered_zscore_normalized.csv"
mine_entry_path = factor_data_directory + "1miEQMineEntryswales_filtered_zscore_normalized.csv"
faults_path = factor_data_directory + "1miEQfaultswales_filtered_zscore_normalized.csv"
dist_faults_path = factor_data_directory + "1miEQdistfaultswales_filtered_zscore_normalized.csv"
breaks_path = factor_data_directory + "1miEQbreaksswales_filtered_zscore_normalized.csv"
rivers_path = factor_data_directory + "1miEQRiverswales_filtered_zscore_normalized.csv"

mine_colleries_factor = pd.read_csv(mine_colleries_path)
mine_entry_factor = pd.read_csv(mine_entry_path)
faults_factor = pd.read_csv(faults_path)
dist_faults_factor = pd.read_csv(dist_faults_path)
breaks_factor = pd.read_csv(breaks_path)
rivers_factor = pd.read_csv(rivers_path)

# Load earthquake data
earthquake_data = pd.read_csv("C:/Users/white/OneDrive/Desktop/Seismology/EQData/eqSouthWales_filtered_zscore_normalized.csv")

# Convert the date column in earthquake data to datetime
earthquake_data['yyyy.mm.dd'] = pd.to_datetime(earthquake_data['yyyy.mm.dd'], format='%d/%m/%Y')

# Convert the date column in factor data to datetime
mine_colleries_factor['yyyy.mm.dd'] = pd.to_datetime(mine_colleries_factor['yyyy.mm.dd'], format='%d/%m/%Y')
mine_entry_factor['yyyy.mm.dd'] = pd.to_datetime(mine_entry_factor['yyyy.mm.dd'], format='%d/%m/%Y')
faults_factor['yyyy.mm.dd'] = pd.to_datetime(faults_factor['yyyy.mm.dd'], format='%d/%m/%Y')
dist_faults_factor['yyyy.mm.dd'] = pd.to_datetime(dist_faults_factor['yyyy.mm.dd'], format='%d/%m/%Y')
breaks_factor['yyyy.mm.dd'] = pd.to_datetime(breaks_factor['yyyy.mm.dd'], format='%d/%m/%Y')
rivers_factor['yyyy.mm.dd'] = pd.to_datetime(rivers_factor['yyyy.mm.dd'], format='%d/%m/%Y')

# Create dataframe for merged data
merged_data = pd.DataFrame()

# Rename columns in factor datasets to avoid duplicate column names
suffixes = ['_mine_colleries', '_mine_entry', '_faults', '_dist_faults', '_breaks', '_rivers']
mine_colleries_factor = mine_colleries_factor.add_suffix(suffixes[0])
mine_entry_factor = mine_entry_factor.add_suffix(suffixes[1])
faults_factor = faults_factor.add_suffix(suffixes[2])
dist_faults_factor = dist_faults_factor.add_suffix(suffixes[3])
breaks_factor = breaks_factor.add_suffix(suffixes[4])
rivers_factor = rivers_factor.add_suffix(suffixes[5])

# Merge the factor datasets into one dataset using 'factor_lat' and 'factor_lon' data
merged_data = pd.merge(earthquake_data, mine_colleries_factor, left_on=['lat', 'lon'], right_on=['lat' + suffixes[0], 'lon' + suffixes[0]], how='left')
merged_data = pd.merge(merged_data, mine_entry_factor, left_on=['lat', 'lon'], right_on=['lat' + suffixes[1], 'lon' + suffixes[1]], how='left')
merged_data = pd.merge(merged_data, faults_factor, left_on=['lat', 'lon'], right_on=['lat' + suffixes[2], 'lon' + suffixes[2]], how='left')
merged_data = pd.merge(merged_data, dist_faults_factor, left_on=['lat', 'lon'], right_on=['lat' + suffixes[3], 'lon' + suffixes[3]], how='left')
merged_data = pd.merge(merged_data, breaks_factor, left_on=['lat', 'lon'], right_on=['lat' + suffixes[4], 'lon' + suffixes[4]], how='left')
merged_data = pd.merge(merged_data, rivers_factor, left_on=['lat', 'lon'], right_on=['lat' + suffixes[5], 'lon' + suffixes[5]], how='left')

# Print the columns of the merged dataframe
print(merged_data.columns)
print(" ")

independent_vars = [
    'ML_mine_colleries',
    'depth_mine_colleries',
    'ML_mine_entry',
    'depth_mine_entry',
    'ML_faults',
    'depth_faults',
    'ML_dist_faults',
    'depth_dist_faults',
    'ML_breaks',
    'depth_breaks'
]

dependent_var = 'ML'  # Change this to your dependent variable column

# Prepare the data for regression
X = merged_data[independent_vars]
y = merged_data[dependent_var]

# Create an instance of SimpleImputer to replace NaN values with the mean
imputer = SimpleImputer()
X_imputed = imputer.fit_transform(X)

# Check for infinite or NaN values in X_imputed
print("Contains Infinite Values:", np.any(np.isinf(X_imputed)))
print("Contains NaN Values:", np.any(np.isnan(X_imputed)))
print(" ")

# If there are no infinite or NaN values, proceed with model fitting
if not (np.any(np.isinf(X_imputed)) or np.any(np.isnan(X_imputed))):
    # Fit the linear regression model without using sample weights
    model = LinearRegression(fit_intercept=False)
    model.fit(X_imputed, y)

    # Predict the target variable
    y_pred = model.predict(X_imputed)

    # Calculate R-squared value
    r_squared = r2_score(y, y_pred)

    # Print the regression results
    print("Coefficients:", model.coef_)
    print("R-squared:", r_squared)
    print(" ")

    # Coefficients and corresponding feature names
    coefficients = model.coef_
    feature_names = [
        'ML_mine_colleries',
        'depth_mine_colleries',
        'ML_mine_entry',
        'depth_mine_entry',
        'ML_faults',
        'depth_faults',
        'ML_dist_faults',
        'depth_dist_faults',
        'ML_breaks',
        'depth_breaks'
    ]

    # Create a bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, coefficients, color='skyblue')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Independent Variables')
    plt.title('Coefficients of Linear Regression Model')
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Set the map center (Wales)
    map_center = [52.5, -3.8]

    # Create a folium map centered on a specific location
    m = folium.Map(location=map_center, zoom_start=7)

    # Create a HeatMap layer using earthquake data
    heat_data = [[row['lat'], row['lon'], row['ML']] for idx, row in earthquake_data.iterrows()]

    # Create a HeatMap layer on the map
    folium.TileLayer('cartodbpositron').add_to(m)  # Add a tile layer for better visibility
    folium.plugins.HeatMap(heat_data).add_to(m)

    # Display the map
    map_file_path = 'C:/Users/white/OneDrive/Desktop/Seismology/EQData/heatmap.html'
    m.save(map_file_path)   

    # Add a constant term to the independent variables matrix for statsmodels
    X_const = sm.add_constant(X_imputed)

    # Fit the linear regression model using statsmodels
    model_stats = sm.OLS(y, X_const).fit()

    # Get the summary of the regression
    summary = model_stats.summary()

    # Print the summary
    print(summary)
else:
    print("X_imputed contains infinite or NaN values. Check your data preprocessing.")


