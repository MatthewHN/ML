import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Employee Salary App", page_icon="💰", layout="wide")


@st.cache_data
def load_model():
    with open("model.pkl", "rb") as f:
        saved = pickle.load(f)
    return saved["model"], saved["job_columns"]


model, job_columns = load_model()

# Derive a clean list of job titles from the dummy columns:
JOB_TITLES = [col.replace("Job Title_", "") for col in job_columns]
# If you want to include the “base” (first category dropped), you can insert it:
# JOB_TITLES.insert(0, "Base Category")

st.title("💰 Employee Salary Prediction")
st.markdown("Select a job title & years of experience to see your predicted salary.")

job = st.selectbox("Job Title", JOB_TITLES)
exp = st.slider("Years of Experience", min_value=0, max_value=40, value=2)

if st.button("Predict"):
    # build one-row DataFrame for the model
    row = {"const": 1, "Years of Experience": exp}
    for col in job_columns:
        row[col] = 1 if col == f"Job Title_{job}" else 0
    X_pred = pd.DataFrame([row])

    sqrt_pred = model.predict(X_pred)[0]
    salary_pred = sqrt_pred ** 2

    st.success(f"💵 Predicted Salary: ${salary_pred:,.0f}")
