#Kalyan
import numpy as np
import pandas as pd
num_samples = 500
df = pd.DataFrame({
    'heart_rate': np.random.randint(60, 130, num_samples),
    'weight': np.random.uniform(45, 120, num_samples).round(1),
    'bmi': np.random.uniform(15, 40, num_samples).round(1),
    'steps_per_day': np.random.randint(1000, 25000, num_samples),
    'sleep_hours': np.random.uniform(3, 10, num_samples).round(1),
    'calories_consumed': np.random.randint(1200, 4500, num_samples),
    'water_intake_liters': np.random.uniform(0.5, 5.0, num_samples).round(1),
    'workout_minutes': np.random.randint(0, 120, num_samples)
})

df['fit_status'] = ['Fit'] * (num_samples // 2) + ['Not Fit'] * (num_samples // 2)
df = df.sample(frac=1).reset_index(drop=True)
csv_path = 'fitness_data.csv'
df.to_csv(csv_path, index=False)
print(f"Dataset created and saved to '{csv_path}'.")
print(df.head())
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