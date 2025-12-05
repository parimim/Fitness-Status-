#pradeeksha
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Reuse preprocessing by importing the script (it standardizes `fitness_data.csv`)
import data_preprocessing

# Read standardized dataset
df = pd.read_csv('fitness_data.csv')

# Ensure fit_status numeric
if df['fit_status'].dtype == object:
    df['fit_status'] = df['fit_status'].map({'Fit': 1, 'Not Fit': 0})

# Feature columns (same order used for interactive input)
feature_cols = [
    'heart_rate', 'weight', 'bmi', 'steps_per_day',
    'sleep_hours', 'calories_consumed', 'water_intake_liters', 'workout_minutes'
]

X = df[feature_cols]   #considering feature columns
y = df['fit_status']   #target variable as fit_status

# Train/test split and train a simple Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Print a single simple metric
acc = accuracy_score(y_test, model.predict(X_test))
print(f"Training complete — accuracy: {acc:.3f}") #need 3 decimal places of accuracy

