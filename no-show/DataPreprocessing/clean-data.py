import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv(
    'no-show/Data/apptdata-no-show part-original.csv')

data.info()

df_copy = data.copy()
if df_copy is None:
    raise ValueError("DataFrame is None")

###################### GENDER########################

df_copy['Gender'].replace('M', 1, inplace=True)
df_copy['Gender'].replace('F', 0, inplace=True)

gender_values = df_copy['Gender'].unique()

# Check if there are only 0 and 1 values
if set(gender_values) == {0, 1}:
    print("The 'gender' column contains only 0 and 1 values.")
else:
    print("The 'gender' column contains values other than 0 and 1.")

###################### WEEKDAY#######################

 # Define a dictionary to map days of the week to numerical values
day_to_number = {
    'Mon': 1,
    'Tue': 2,
    'Wed': 3,
    'Thu': 4,
    'Fri': 5,
    'Sat': 6,
    'Sun': 7
}

# Apply mapping to the 'day_of_week' column
df_copy['weekday'] = df_copy['weekday'].map(day_to_number)

day_of_week_values = df_copy['weekday'].unique()

# Check if there are only values from one to seven
if set(day_of_week_values) == set(range(1, 8)):
    print("The 'day of the week' column contains values from one to seven.")
else:
    print("The 'day of the week' column contains values other than one to seven.")


###################### Visit Type#######################

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the 'visit_type' column
df_copy['Visit Type'] = label_encoder.fit_transform(df_copy['Visit Type'])


###################### Age#######################

# Check for negative age values
negative_age = df_copy[df_copy['Age'] < 0]

# Display rows with negative age values
if not negative_age.empty:
    print("Rows with negative age values:")
    print(negative_age)
else:
    print("No rows with negative age values.")

###################### Lead time#######################
df_copy['lead time'].unique()
df_copy['lead time'] = df_copy['lead time'].apply(lambda x: x if x >= 0 else 0)
df_copy['lead time'].unique()


###################### Missing Values#######################

print("Original DataFrame shape:", df_copy.shape)

# Count missing values in each row
missing_values_per_row = df_copy.isnull().sum(axis=1)
print("Columns with Missing Values:")
print(missing_values_per_row[missing_values_per_row > 0])

# Filter rows where the count of missing values is greater than zero
rows_to_remove = missing_values_per_row[missing_values_per_row > 0].index

# Drop rows with missing values
df = df_copy.drop(rows_to_remove)

# Print the original DataFrame shape and the cleaned DataFrame shape
print("New DataFrame shape:", df.shape)

df_copy = df

# # Handle missing values
# df_copy['event'].fillna('', inplace=True)

scaler = MinMaxScaler()
df_copy[['Visit Type', 'lead time', 'weekday', 'Age']] = scaler.fit_transform(
    df_copy[['Visit Type', 'lead time', 'weekday', 'Age']])

###################### Clean Column#######################
# Remove Column
df_copy = df_copy.drop('Provider ID', axis=1)
df_copy = df_copy.drop('Patient ID', axis=1)
df_copy = df_copy.drop('Hep', axis=1)
df_copy = df_copy.drop('event', axis=1)
df_copy = df_copy.drop('No Show %', axis=1)


df_copy['R_refill/Referral'] = df_copy['Referral'] | df_copy['refill']
df_copy['R_Annual_Exam'] = df_copy['Annual GYN Exam'] | df_copy['routine physic/ spor']
df_copy = df_copy.drop('Referral', axis=1)
df_copy = df_copy.drop('refill', axis=1)
df_copy = df_copy.drop('Annual GYN Exam', axis=1)
df_copy = df_copy.drop('routine physic/ spor', axis=1)


###################### Rename Column#######################
df_copy.rename(columns={'Visit Type': 'Visit_Type'}, inplace=True)
df_copy.rename(columns={'weekday': 'WeekDay'}, inplace=True)
df_copy.rename(columns={'lead time': 'Lead_Time'}, inplace=True)
df_copy.rename(columns={'patient type': 'Patient_Type'}, inplace=True)
df_copy.rename(columns={'outcome': 'Outcome'}, inplace=True)
df_copy.rename(columns={'numberofreasons': 'Number_Of_Reasons'}, inplace=True)
df_copy.rename(columns={'vision': 'R_Vision'}, inplace=True)
df_copy.rename(columns={'sexual health': 'R_Sexual_Health'}, inplace=True)
df_copy.rename(columns={'respiratory': 'R_Respiratory'}, inplace=True)
df_copy.rename(columns={'PPD/PPD READING': 'R_PPD/PPD_reading'}, inplace=True)
df_copy.rename(columns={'orthopidic': 'R_Orthopedic'}, inplace=True)
df_copy.rename(columns={'mental health': 'R_Mental_Health'}, inplace=True)
df_copy.rename(columns={'Immunization': 'R_Immunization'}, inplace=True)
df_copy.rename(columns={'gastropathy': 'R_Gastropathy'}, inplace=True)
df_copy.rename(columns={'emergency': 'R_Emergency'}, inplace=True)
df_copy.rename(columns={'dermatology': 'R_Dermatology'}, inplace=True)
df_copy.rename(columns={'Cancer': 'R_Cancer'}, inplace=True)
df_copy.rename(columns={'STD': 'R_STD'}, inplace=True)
df_copy.rename(columns={'Cholestrol': 'R_Cholesterol'}, inplace=True)
df_copy.rename(columns={'Diabetes': 'R_Diabetes'}, inplace=True)
df_copy.rename(columns={'Surgery': 'R_Surgery'}, inplace=True)
df_copy.rename(columns={'Concussion': 'R_Concussion'}, inplace=True)
df_copy.rename(columns={'chronic_count': 'Chronic_Count'}, inplace=True)
df_copy.rename(columns={'Asthma': 'C_Asthma'}, inplace=True)
df_copy.rename(columns={'Alcoholic': 'C_Alcoholic'}, inplace=True)
df_copy.rename(columns={'Anxiety': 'C_Anxiety'}, inplace=True)
df_copy.rename(columns={'BP': 'C_Blood_Pressure'}, inplace=True)
df_copy.rename(columns={'Heart': 'C_Heart_Problem'}, inplace=True)
df_copy.rename(columns={'Smoker': 'C_Smoker'}, inplace=True)
df_copy.rename(columns={'Seizures': 'C_Seizures'}, inplace=True)
df_copy.rename(
    columns={'Learning_disabilities': 'C_Learning_Disability'}, inplace=True)
df_copy.rename(columns={'F_binary': 'C_Family_anxiety'}, inplace=True)
df_copy.rename(columns={'#ofvaccsin': 'Number_of_vaccines'}, inplace=True)

df_copy.rename(columns={'binary_outcome': 'Binary_Outcome'}, inplace=True)
df_copy.rename(columns={'W_event': 'Number_Of_Events'}, inplace=True)


###################### Sort#######################
column_priority = ['Binary_Outcome', 'Outcome', 'Patient_Type', 'Gender', 'Age', 'WeekDay', 'Number_Of_Events', 'Lead_Time',
                   'Visit_Type', 'Chronic_Count', 'Number_Of_Reasons', 'R_Vision', 'R_Annual_Exam',
                   'R_Sexual_Health', 'R_Respiratory', 'R_PPD/PPD_reading', 'R_Orthopedic', 'R_Mental_Health',
                   'R_Immunization', 'R_Gastropathy', 'R_Emergency', 'R_Dermatology', 'R_Cancer', 'R_STD',
                   'R_Cholesterol', 'R_Diabetes', 'R_Surgery', 'R_Concussion', 'R_refill/Referral', 'C_Asthma', 'C_Alcoholic',
                   'C_Anxiety', 'C_Blood_Pressure', 'C_Heart_Problem', 'C_Smoker', 'C_Seizures',
                   'C_Learning_Disability', 'C_Family_anxiety', 'Number_of_vaccines']

# Get the sorted columns
sorted_columns = sorted(df_copy.columns, key=lambda x: column_priority.index(
    x) if x in column_priority else -1)

# Create the reordered DataFrame
df_copy = df_copy[sorted_columns]


column_count2 = df_copy.shape[1]
print("Number of columns:", column_count2)
print("Columns are:", df_copy.columns)
print("Columns are:", data.columns)


# Save the preprocessed DataFrame to a CSV file
df_copy.to_csv(
    'no-show/DataPreprocessing/no-show-cleaned-data.csv', index=False)

df_copy.describe()

df_copy.info()
