# Factor Analysis Ver 5
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

# Create lists to store dates and R-squared values
dates = []
r_squared_values = []

# Perform temporal analysis
time_intervals = pd.date_range(start=merged_data['yyyy.mm.dd'].min(), end=merged_data['yyyy.mm.dd'].max(), freq='M')

for start_date, end_date in zip(time_intervals, time_intervals[1:]):
    interval_data = merged_data[(merged_data['yyyy.mm.dd'] >= start_date) & (merged_data['yyyy.mm.dd'] < end_date)]

    if interval_data.shape[0] < 2:
        print(f"Skipping interval with insufficient data: {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")
        print("=" * 80)
        continue

    # Prepare the data for regression
    X = interval_data[independent_vars]
    y = interval_data[dependent_var]

    if not X.empty:
        # Remove columns with all missing values
        X = X.dropna(axis=1, how='all')

        if X.shape[1] == 0:
            print(f"No valid independent variables available for {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")
            print("=" * 80)
            continue

        # Create an instance of SimpleImputer to replace NaN values with the mean
        imputer = SimpleImputer()
        X_imputed = imputer.fit_transform(X)

        # Fit the linear regression model without using sample weights
        model = LinearRegression(fit_intercept=False)
        model.fit(X_imputed, y)

        # Coefficients and corresponding feature names
        coefficients = model.coef_
        feature_names = X.columns.tolist()

        # Add a constant term to the independent variables matrix for statsmodels
        X_const = sm.add_constant(X_imputed)

        # Fit the linear regression model using statsmodels
        model_stats = sm.OLS(y, X_const).fit()

        # Get the summary of the regression
        print(f"Temporal Analysis for {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}:")
        print(model_stats.summary())
        print("=" * 80)

        # Store R-squared value for visualization
        dates.append(end_date)
        r_squared_values.append(model_stats.rsquared)

    else:
        print(f"No valid independent variables available for {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")
        print("=" * 80)

# Create a line plot for R-squared values over time intervals
plt.figure(figsize=(10, 6))
plt.plot(dates, r_squared_values, marker='o')
plt.xlabel('Time Interval')
plt.ylabel('R-squared Value')
plt.title('Temporal Analysis of R-squared Values')
plt.xticks(rotation=45)
plt.tight_layout()

# Display the plot
plt.show()
