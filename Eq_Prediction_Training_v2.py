#########################################################################################################################################################################
#                                                                                                                                                                       #
#                                                       Earthquake Potential Prediction Training Program Ver 1.2                                                        #
#                                                                                                                                                                       #
#########################################################################################################################################################################
#                                                                                                                                                                       #  
#                                                                   Created by                                                                                          #
#                                                                   Chris Harrison                                                                                      #
#                                                                   06/08/2023                                                                                          #
#                                                                                                                                                                       #
#########################################################################################################################################################################
#                                                                                                                                                                       #
# Python program created to apply a neural network to predict potential earthquake locations. Data has been extracted from the British Geological Society Website.      #
# The main dataset contains the earthquake data from the South Wales region. The seven factor dataset have are recorded extracted from the South Wales region dataset   #
# that are within 1 mile of different factors related to the South Wales coalfield. Factors such as Fault lines, Mine and Collery locations, Rivers etc.                #
# The applications uses the data to create a training and test set to learn how to predict potential locations. A pickle (.PKL), and .h5 file are created which is      #
# used in another application to be applied to a mucvh larger dataset. The .h5 file seems to be more stable to use on this other dataset.                               #
#                                                                                                                                                                       #
#########################################################################################################################################################################



# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import folium
from folium import plugins

# Set enviroment variables and suppress warnings
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "14" # Computing deviced used to create the application has a max of 14 cores, adjust based on your own device
import warnings
warnings.filterwarnings("ignore")


# Load the main earthquake dataset
main_dataset_path = "C:/Users/white/OneDrive/Desktop/Seismology/EQData/Correlation/Cluster Analysis 1/Prediction/eqSouthWalesrefine.csv"
main_dataset = pd.read_csv(main_dataset_path)

# Set random seed for reproductibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the derived datasets that 1 mile from a factor - Faultline, rivers, lakes, mine and collery locations etc.
derived_datasets_directory = "C:/Users/white/OneDrive/Desktop/Seismology/EQData/Correlation/Cluster Analysis 1/Prediction/"
mine_colleries_path = derived_datasets_directory + "1miEQMineColleriesswales.csv"
mine_entry_path = derived_datasets_directory + "1miEQMineEntryswales.csv"
faults_path = derived_datasets_directory + "1miEQfaultswales.csv"
dist_faults_path = derived_datasets_directory + "1miEQdistfaultswales.csv"
breaks_path = derived_datasets_directory + "1miEQbreaksswales.csv"
lakes_path = derived_datasets_directory + "1miEQLakesswales.csv"
rivers_path = derived_datasets_directory + "1miEQRiverswales.csv"

mine_colleries_data = pd.read_csv(mine_colleries_path)
mine_entry_data = pd.read_csv(mine_entry_path)
faults_data = pd.read_csv(faults_path)
dist_faults_data = pd.read_csv(dist_faults_path)
breaks_data = pd.read_csv(breaks_path)
lakes_data = pd.read_csv(lakes_path)
rivers_data = pd.read_csv(rivers_path)

# Set columns as the index
mine_colleries_data.set_index(['Nsta', 'RMS', 'intensity', 'induced'], inplace=True)
mine_entry_data.set_index(['Nsta', 'RMS', 'intensity', 'induced'], inplace=True)
faults_data.set_index(['Nsta', 'RMS', 'intensity', 'induced'], inplace=True)
dist_faults_data.set_index(['Nsta', 'RMS', 'intensity', 'induced'], inplace=True)
breaks_data.set_index(['Nsta', 'RMS', 'intensity', 'induced'], inplace=True)
lakes_data.set_index(['Nsta', 'RMS', 'intensity', 'induced'], inplace=True)
rivers_data.set_index(['Nsta', 'RMS', 'intensity', 'induced'], inplace=True)


# Merge the factor datasets into one dataset using 'lat' and 'lon' data
merged_data = pd.merge(main_dataset, mine_colleries_data, on=['lat', 'lon'], how='left', suffixes=('_main', '_mine_colleries'))
merged_data = pd.merge(merged_data, mine_entry_data, on=['lat', 'lon'], how='left', suffixes=('_mine_colleries', '_mine_entry'))
merged_data = pd.merge(merged_data, faults_data, on=['lat', 'lon'], how='left', suffixes=('_mine_entry', '_faults'))
merged_data = pd.merge(merged_data, dist_faults_data, on=['lat', 'lon'], how='left', suffixes=('_faults', '_dist_faults'))
merged_data = pd.merge(merged_data, breaks_data, on=['lat', 'lon'], how='left', suffixes=('_dist_faults', '_breaks'))
merged_data = pd.merge(merged_data, lakes_data, on=['lat', 'lon'], how='left', suffixes=('_breaks', '_lakes'))
merged_data = pd.merge(merged_data, rivers_data, on=['lat', 'lon'], how='left', suffixes=('_lakes', '_rivers'))

# Feature Engineering
# Process and convert the features into numerical representations
# For simplicity, let's assume you selected some relevant columns from each dataset
selected_columns = ['depth_main', 'ML_main', 'Nsta', 'RMS', 'intensity', 'induced']

# Create the binary target column, threshold_magnitude set at mid-point value of ML from the main earthquake set
threshold_magnitude = 2.5 # adjust threshold_magnitude, however setting to 4 results in no True Values
merged_data['potential_earthquake'] = (merged_data['ML_main'] >= threshold_magnitude).astype(int)

# Remove unnecessary columns and fill missing values with zeros
merged_data[selected_columns] = merged_data[selected_columns].apply(pd.to_numeric, errors='coerce')  # Convert to numeric
merged_data.fillna(0, inplace=True)  # Fill remaining missing values with zeros

# Split the data into features (X) and the binary target (y)
X = merged_data[selected_columns]
y = merged_data['potential_earthquake']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalise numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply Random Under-Sampling to balance the classes
under_sampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)  # Adjust the sampling_strategy based on your desired class ratio
X_train_scaled_resampled, y_train_resampled = under_sampler.fit_resample(X_train_scaled, y_train)

# Apply Synthetic Minority Over-sampling Technique (SMOTE) to balance the classes
k_neighbors = 5  # Set the number of neighbors for SMOTE
smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
X_train_scaled_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)


# Build a neural network model for earthquake prediction
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(X_train_scaled.shape[1],)))  # Increase neurons
model.add(Dropout(0.5))  # Increase dropout for regularization
model.add(Dense(512, activation='relu'))  # Add another hidden layer
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))  # Add another hidden layer
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))  # Add another hidden layer
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))  # Add another hidden layer
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))


class_weights = compute_class_weight('balanced', classes=np.unique(y_train_resampled), y=y_train_resampled)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train_resampled), class_weights)}
# Compile the model with appropriate loss function and optimizer
model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

# Add a learning rate scheduler callback to adapt learning rate during training
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_lr=1e-7)

# Train the model with the resampled data and learning rate scheduler callback
class_weights = {0: 1, 1: 5}  # Adjust the class weights based on your data
history = model.fit(X_train_scaled_resampled, y_train_resampled, epochs=100, batch_size=32, validation_split=0.2, callbacks=[lr_scheduler], class_weight=class_weights)


# Evaluate the trained model using the test set
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Code section used to test and visual trained model

# Make earthquake predictions using the trained model using the test set
predictions = model.predict(X_test_scaled)

# Convert predictions to a DataFrame
results_df = pd.DataFrame({
    'True_Values': y_test.values, 
    'Predictions': predictions.flatten()
})

# Save the results dataframe to a CSV file
#results_csv_path = "C:/Users/white/OneDrive/Desktop/Seismology/EQData/Correlation/Cluster Analysis 1/predictions.csv"
#results_df.to_csv(results_csv_path, index=False)


# Convert predictions to a DataFrame
prediction_df = pd.DataFrame(predictions, columns=['Predictions'])

# Create a DataFrame to store the predictions and true values
results_df = pd.DataFrame({
    'True_Values': y_test,           
    'Predictions': prediction_df['Predictions']  
})

# Save the DataFrame to a CSV file
#results_df.to_csv('C:/Users/white/OneDrive/Desktop/Seismology/EQData/Correlation/Cluster Analysis 1/predictions_results.csv', index=False)

#  Visualise predictions against true values to assess model performance
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.plot([0, 1], [0, 1], 'r--', lw=3)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs. Predictions')
plt.show()

# Make predictions using the model
predictions = model.predict(X_test_scaled)

# Convert predictions to a DataFrame
prediction_df = pd.DataFrame(predictions, columns=['Predictions'])

# Create a DataFrame to store the predictions and true values
results_df = pd.DataFrame({
    'True_Values': y_test,           # Replace 'y_test' with the true target values from the test set
    'Predictions': prediction_df['Predictions']  # Replace 'prediction_df' with the DataFrame containing your predictions
})

#print(X_test.columns)

# Set a threshold for identifying potential earthquake locations
threshold = 0.005  # You can adjust this threshold based on your confidence level

# Store the columns needed for potential locations in a separate DataFrame
potential_columns = ['lat', 'lon']  # Only include lat and lon for potential locations
potential_data = main_dataset[potential_columns]

# Get the indices of potential earthquake locations
potential_locations_indices = np.where(results_df['Predictions'] > threshold)[0]

# Check if there are any potential locations
if len(potential_locations_indices) > 0:
    # Extract the potential earthquake locations using iloc
    potential_locations = potential_data.iloc[potential_locations_indices]

    # Save the potential locations to a CSV file
    potential_locations.to_csv('C:/Users/white/OneDrive/Desktop/Seismology/EQData/Correlation/Cluster Analysis 1/potential_earthquake_locations.csv', index=False)
else:
    print("No potential earthquake locations were identified.")
    # Create an empty DataFrame with the desired columns
    empty_data = {col: [] for col in potential_columns}
    potential_locations = pd.DataFrame(empty_data)
    # Save the empty DataFrame to a CSV file
    potential_locations.to_csv('C:/Users/white/OneDrive/Desktop/Seismology/EQData/Correlation/Cluster Analysis 1/potential_earthquake_locations.csv', index=False)
    
# Visualise the predictions against the true values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.plot([0, 1], [0, 1], 'r--', lw=3)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs. Predictions')
plt.show()

# Visualise the distribution of predicted probabilities
plt.hist(predictions, bins=20, edgecolor='k')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Probabilities')
plt.show()

# Make predictions using the model
y_pred = model.predict(X_test_scaled)

# Convert predictions to binary labels based on the threshold (0.0054 in this case)
threshold = 0.5
y_pred_binary = (y_pred > threshold).astype(int)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)

# Calculate performance metrics using predictions from the model
y_pred = model.predict(X_test_scaled)
y_pred_binary = (y_pred > threshold).astype(int)

# Handle ROC AUC score when there is only one class in the test set
try:
    roc_auc = roc_auc_score(y_test, y_pred)
except ValueError:
    roc_auc = 1.0

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("ROC AUC:", roc_auc)

# Calculate Average Precision (AUC-PRC) as an additional evaluation metric
average_precision = average_precision_score(y_test, y_pred)
print("Average Precision:", average_precision)

# Create an interactive map to visualize potential earthquake locations

# Load the file named 'potential_earthquakes.csv'
file_path = 'C:/Users/white/OneDrive/Desktop/Seismology/EQData/Correlation/Cluster Analysis 1/potential_earthquake_locations.csv'
potential_earthquakes = pd.read_csv(file_path)

# Set the map center (Wales)
map_center = [52.5, -3.8]

# Create a heatmap to visualize potential earthquake locations

# Create a new Folium map
map_osm = folium.Map(location=map_center, zoom_start=7)

# Convert potential earthquake coordinates to a list of points
heatmap_data = [[row['lat'], row['lon']] for index, row in potential_earthquakes.iterrows()]

# Create a HeatMap layer on the map
folium.TileLayer('cartodbpositron').add_to(map_osm)  # Add a tile layer for better visibility
folium.plugins.HeatMap(heatmap_data).add_to(map_osm)

# Save the heatmap map as an HTML file
heatmap_file_path = 'C:/Users/white/OneDrive/Desktop/Seismology/EQData/Correlation/Cluster Analysis 1/potential_earthquakes_heatmap.html'
map_osm.save(heatmap_file_path)
