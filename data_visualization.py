#omm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset (preprocessed by data_preprocessing.py)
csv_path = 'fitness_data.csv'
print(f"Loading dataset from '{csv_path}' for visualization...")
df = pd.read_csv(csv_path)

# If fit_status is numeric, map back to labels for clearer plots
if df['fit_status'].dtype != object:
    df['fit_status'] = df['fit_status'].map({1: 'Fit', 0: 'Not Fit'})

# Count plot: Fit vs Not Fit
plt.figure()
sns.countplot(x='fit_status', data=df)
plt.title("Count of Fit vs Not Fit")
plt.xlabel("Fit Status")
plt.ylabel("Count")
plt.grid(True)
plt.show()

# BMI distribution
plt.figure()
plt.hist(df['bmi'], bins=20)
plt.title("BMI Distribution")
plt.xlabel("BMI")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()




#sailaja
# Sleep hours comparison (boxplot)
plt.figure(figsize=(7, 5))
sns.boxplot(x='fit_status', y='sleep_hours', data=df)
plt.title("Sleep Hours Comparison Between Fit and Not Fit")
plt.xlabel("Fitness Status")
plt.ylabel("Sleep Hours")
plt.show()

# Scatter: steps vs workout
plt.figure()
sns.scatterplot(x='steps_per_day', y='workout_minutes', hue='fit_status', data=df)
plt.title("Workout Minutes vs Steps Per Day")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="RdBu", linewidths=0.5)
plt.title("Correlation Heatmap of Health Dataset", fontsize=16)
plt.show()

print("Visualizations complete.")
