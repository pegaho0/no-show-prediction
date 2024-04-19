import pandas as pd
from imblearn.over_sampling import RandomOverSampler


# Read the CSV file into a DataFrame
data = pd.read_csv(
    'no-show/VariableSelection/no-show-selected-features.csv')

# Assuming your target variable is in a column named 'target'
X = data.drop(columns=["Binary_Outcome"])
y = data["Binary_Outcome"]

# Specify the column names for the resampled DataFrame
column_names = X.columns

# Initialize RandomOverSampler
ros = RandomOverSampler(random_state=42)

# Perform Random Over Sampling
X_resampled, y_resampled = ros.fit_resample(X, y)

# Convert resampled data to DataFrame
df_resampled = pd.DataFrame(X_resampled, columns=column_names)
df_resampled['Binary_Outcome'] = y_resampled

# Save resampled data to CSV file
df_resampled.to_csv(
    'no-show/DataBalancing/ROS_resampled_data.csv', index=False)

# Print the class distribution after ROS
print("Class distribution after Random Under Sampling:")
print(y_resampled.value_counts())
