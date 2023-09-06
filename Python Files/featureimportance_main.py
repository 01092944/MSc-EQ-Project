# Feature Importance Analysis
# Written by Christopher Harrison
# Created on 10/08/2023

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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

# Create a list to store feature importances
feature_importances_list = []

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

    # Create and train a Random Forest Regressor model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Get feature importances from the trained model
    feature_importances = {
        'Dataset': csv_name_mapping[csv_file],
        'ML Importance': model.feature_importances_[0],
        'Depth Importance': model.feature_importances_[1]
    }

    # Append feature importances to the list
    feature_importances_list.append(feature_importances)

# Create a DataFrame from the feature importances list
feature_importances_df = pd.DataFrame(feature_importances_list)

# Print the feature importances
print("Feature Importances:")
print(feature_importances_df)

# Save the feature importances to a CSV file
feature_importances_df.to_csv("feature_importances.csv", index=False)

# Plot the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importances_df, x='Dataset', y='ML Importance', color='blue', label='ML Importance')
sns.barplot(data=feature_importances_df, x='Dataset', y='Depth Importance', color='orange', label='Depth Importance')
plt.title("Feature Importances")
plt.xlabel("Dataset")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

