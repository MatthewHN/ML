import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Employee Salary App", page_icon="💰", layout="wide")

# ─── Load & Train Model (cached) ──────────────────────────────────────────────
@st.cache_data
def load_and_train_model(data_path: str):
    # 1) Load
    df = pd.read_csv(data_path)
    # 2) One-hot encode Job Title
    df = pd.get_dummies(df, columns=["Job Title"], drop_first=True)
    job_cols = [c for c in df.columns if c.startswith("Job Title_")]
    # 3) Features & target (√ Salary)
    X = df[["Years of Experience"] + job_cols]
    y = np.sqrt(df["Salary"].values)
    # 4) Train a simple linear model
    model = LinearRegression().fit(X, y)
    return model, job_cols

model, job_columns = load_and_train_model("Salary_Data - Copy2.csv")

# Derive your clean list of job titles
JOB_TITLES = [col.replace("Job Title_", "") for col in job_columns]

# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("💰 Employee Salary Prediction")
st.markdown("Select a job title & years of experience to see your predicted salary.")

job = st.selectbox("Job Title", JOB_TITLES)
exp = st.slider("Years of Experience", min_value=0, max_value=40, value=2)

if st.button("Predict"):
    # build one‐row input
    row = {"Years of Experience": exp}
    for c in job_columns:
        row[c] = 1 if c == f"Job Title_{job}" else 0
    X_pred = pd.DataFrame([row])
    sqrt_pred = model.predict(X_pred)[0]
    salary_pred = sqrt_pred ** 2
    st.success(f"💵 Predicted Salary: ${salary_pred:,.0f}")
