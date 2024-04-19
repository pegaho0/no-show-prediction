import random
from deap import base, creator, tools, algorithms
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import chi2_contingency
import random
from deap import base, creator, tools, algorithms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

df = pd.read_csv(
    '/home/pegah/Desktop/No-show-Final/Source/no-show/DataPreprocessing/no-show-cleaned-data.csv')

df.info()

df_copy = df.copy()
if df_copy is None:
    raise ValueError("DataFrame is None")


columns_object = df.drop(columns=['Binary_Outcome'], axis=1)
significant_vars = []
significant_vars_list = []

# Iterate over categorical variables (excluding the target variable)
for col in columns_object:
    # Create a contingency table
    contingency_table = pd.crosstab(df[col], df['Binary_Outcome'])

    # Perform chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    # Check if p-value is less than 0.25
    if p < 0.25:
        significant_vars.append((col, p))
        significant_vars_list.append(col)


# Print significant categorical variables and their p-values
print(
    f"Significant categorical variables and their p-values: {len(significant_vars)}")
for var, p_value in significant_vars:
    print(f"{var}: p-value = {p_value}")


# Separate features and target variable
X = df.drop(columns=['Binary_Outcome'], axis=1)
y = df['Binary_Outcome']

# Define evaluation function


def evaluate(individual, X, y):
    # Select features based on individual's binary encoding
    selected_features = [feature for feature,
                         sel in zip(X.columns, individual) if sel]
    X_selected = X[selected_features]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.3, random_state=42)

    # Train a classifier (Random Forest in this example)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predict on the test set and calculate accuracy
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy,


# Define genetic algorithm parameters
POPULATION_SIZE = 10
GENERATIONS = 5
CROSSOVER_PROB = 0.5
MUTATION_PROB = 0.2

# Define types for individuals (binary encoding) and fitness
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Initialize toolbox
toolbox = base.Toolbox()

# Register evaluation function, individual generator, and operators
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat,
                 creator.Individual, toolbox.attr_bool, n=len(X.columns))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate, X=X, y=y)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# Create initial population
population = toolbox.population(n=POPULATION_SIZE)

# Run genetic algorithm
for gen in range(GENERATIONS):
    offspring = algorithms.varAnd(
        population, toolbox, cxpb=CROSSOVER_PROB, mutpb=MUTATION_PROB)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

# Get the best individual from the final population
best_individual = tools.selBest(population, k=1)[0]

# Print the selected features and their corresponding fitness (accuracy)
selected_features = [feature for feature,
                     sel in zip(X.columns, best_individual) if sel]
accuracy = evaluate(best_individual, X, y)[0]
print("Best Individual (Selected Features):", selected_features)
print("Accuracy:", accuracy)

# Convert lists to sets
set1 = set(significant_vars_list)
set2 = set(selected_features)
# Find common elements using intersection
new_decisions = set1.intersection(set2)

# Convert the result back to a list if needed
print("significant_vars_list", selected_features)
# new_decisions.append('Binary_Outcome')


# Convert the set to a list
new_decisions_list = list(new_decisions)

# Append the string to the list
new_decisions_list.append('Binary_Outcome')

# Convert the list back to a set if needed
new_decisions = set(new_decisions_list)

print("new_decisions:", new_decisions)


# Drop the columns that are not in the list
columns_to_drop = [
    col for col in df_copy.columns if col not in new_decisions]
df_copy.drop(columns=columns_to_drop, inplace=True)

# Save the preprocessed DataFrame to a CSV file
df_copy.to_csv(
    '/home/pegah/Desktop/No-show-Final/Source/no-show/VariableSelection/no-show-selected-features.csv', index=False)

df_copy.describe()

df_copy.info()
