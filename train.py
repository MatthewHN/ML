import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

# --- 1) Load your data ---
df = pd.read_csv("Salary_Data - Copy2.csv")

# --- 2) One-hot encode job titles (drop_first to avoid dummy trap) ---
df = pd.get_dummies(df, columns=["Job Title"], drop_first=True, dtype=int)
job_cols = [c for c in df.columns if c.startswith("Job Title_")]

# --- 3) Transform the target ---
y = np.sqrt(df["Salary"].values)

# --- 4) Build the feature matrix ---
X = df[["Years of Experience"] + job_cols]

# --- 5) Fit Random Forest ---
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# --- (Optional) Quick sanity‐check of in‐sample performance ---
from sklearn.metrics import r2_score, mean_squared_error
preds = rf.predict(X)
print(f"R² (train) = {r2_score(y, preds):.3f}, MSE (train) = {mean_squared_error(y, preds):.3f}")

# --- 6) Persist both model & the list of job dummies ---
with open("model.pkl", "wb") as f:
    pickle.dump({"model": rf, "job_columns": job_cols}, f)

print("Random Forest trained and saved to model.pkl.")
