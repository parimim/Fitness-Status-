#Mounika 

import pandas as pd

# Load dataset saved by data_creation.py
csv_path = 'fitness_data.csv'
print(f"Loading dataset from '{csv_path}'...")
df = pd.read_csv(csv_path)

# Check missing values
print("Missing values before preprocessing:\n", df.isnull().sum())

print("\nFirst few rows before preprocessing:")
print(df.head())

# Convert fit_status from text to numeric: Fit -> 1, Not Fit -> 0
# Only map if values are strings (to avoid remapping numeric values)
if df['fit_status'].dtype == object:
    df['fit_status'] = df['fit_status'].map({'Fit': 1, 'Not Fit': 0})

print("\nValue counts for fit_status after mapping:")
print(df['fit_status'].value_counts())

print("\nData description:")
print(df.describe())

# Overwrite the same CSV so all other scripts use this standardized file
df.to_csv(csv_path, index=False)
print(f"\nPreprocessing complete. Saved standardized data to '{csv_path}'.")
