import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle

# --- 1) Load your data ---
# Make sure you've placed your cleaned CSV here (only Salary, Years of Experience, Job Title)
df = pd.read_csv("Salary_Data - Copy2.csv")

# --- 2) One-hot encode job titles (drop_first to avoid dummy trap) ---
df = pd.get_dummies(df, columns=["Job Title"], drop_first=True, dtype=int)
job_cols = [c for c in df.columns if c.startswith("Job Title_")]

# --- 3) Transform the target (you used sqrt in your notebook) ---
y = np.sqrt(df["Salary"].values)

# --- 4) Build the design matrix ---
X = sm.add_constant(df[["Years of Experience"] + job_cols])

# --- 5) Fit OLS ---
model = sm.OLS(y, X).fit()

# --- 6) Persist both model & the list of job dummies ---
with open("model.pkl", "wb") as f:
    pickle.dump({"model": model, "job_columns": job_cols}, f)

print(model.summary())
