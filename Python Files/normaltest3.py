# Feature Linear Regression program Version 1
# Written by Christopher Harrison
# Created on 10/08/2023

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

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

# Create an empty DataFrame to store coefficients
coefficients_df = pd.DataFrame(columns=['Dataset', 'Intercept', 'ML Coefficient', 'Depth Coefficient'])

# Iterate through each CSV file
for csv_file in csv_files:
    # Read the dataset
    data = pd.read_csv(csv_file)

    # Impute missing values using mean imputation
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(data[['ML', 'depth']])

    # Convert imputed data back to DataFrame
    data_imputed_df = pd.DataFrame(data_imputed, columns=['ML', 'depth'])

    # Select the features and target variable
    X = data_imputed_df[['ML', 'depth']]
    y = data_imputed_df['ML']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Store the dataset in the all_data DataFrame
    all_data[csv_name_mapping[csv_file]] = data['ML']

    # Visualize the data
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x='depth', y='ML')
    plt.title(f'{csv_name_mapping[csv_file]} - Data Visualization')
    plt.xlabel('Depth')
    plt.ylabel('ML')
    plt.show()

    # Store coefficients in coefficients_df
    coefficients_df.loc[len(coefficients_df)] = [
        csv_name_mapping[csv_file],
        model.intercept_,
        model.coef_[0],
        model.coef_[1]
    ]
     
# Print the coefficients
print("Linear Regression Coefficients:")
print(coefficients_df)

# Save the coefficients to a CSV file
coefficients_df.to_csv("linear_regression_coefficients.csv", index=False)
