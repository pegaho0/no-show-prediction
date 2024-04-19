import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

# Read the CSV file into a DataFrame
data = pd.read_csv(
    '/home/pegah/Desktop/No-show-Final/Source/no-show/VariableSelection/no-show-selected-features.csv')

# Assuming your target variable is in a column named 'target'
X = data.drop(columns=["Binary_Outcome"])
y = data["Binary_Outcome"]

# Specify the column names for the resampled DataFrame
column_names = X.columns

# Initialize RandomunderSampler
rus = RandomUnderSampler(random_state=42)

# Perform Random Under Sampling
X_resampled, y_resampled = rus.fit_resample(X, y)

# Convert resampled data to DataFrame
df_resampled = pd.DataFrame(X_resampled, columns=column_names)
df_resampled['Binary_Outcome'] = y_resampled

# Save resampled data to CSV file
df_resampled.to_csv(
    '/home/pegah/Desktop/No-show-Final/Source/no-show/DataBalancing/RUS_resampled_data.csv', index=False)

# Print the class distribution after ROS
print("Class distribution after Random Under Sampling:")
print(y_resampled.value_counts())
