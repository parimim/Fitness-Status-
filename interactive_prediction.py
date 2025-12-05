#Sowmya
import pandas as pd
import model_training

features = model_training.feature_cols
model = model_training.model

vals = []
for feat in features:
    v = input(f"{feat}: ")
    try:
        vals.append(float(v))
    except:
        vals.append(0.0)

user_df = pd.DataFrame([vals], columns=features)
pred = model.predict(user_df)[0]
if pred == 1:   
    print("Predicted Fitness Status: Fit")
else:
    print("Predicted Fitness Status: Not Fit")
