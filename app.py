import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from streamlit_option_menu import option_menu


# --- Page Config & CSS ---
st.set_page_config(page_title="Employee Salary App by Matthew Horne", page_icon="💰", layout="wide")

# --- Load Data & Model ---
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data
def load_model(path: str):
    with open(path, "rb") as f:
        saved = pickle.load(f)
    return saved["model"], saved["job_columns"]

# Paths
DATA_PATH = "Salary_Data.csv"  # your CSV with columns: Salary, Years of Experience, Job Title
MODEL_PATH = "model.pkl"  # your pickled model dict {model, job_columns}

# Load
df = load_data(DATA_PATH)
model, job_columns = load_model(MODEL_PATH)
JOB_TITLES = [col.replace("Job Title_", "") for col in job_columns]

st.markdown(
    """
    <style>
    /* Sidebar */
    div[data-testid="stSidebar"] {
        background-color: #111;
        border-right: 4px solid #222;
        padding: 1rem;
    }
    /* Hide header/footer */
    #MainMenu, footer { visibility: hidden; }
    /* Select box styling */
    div[data-baseweb="select"] > div {
        background-color: #111;
        border: 2px solid #7B06A6;
        color: white;
    }
    /* Slider styling */
    div[data-baseweb="range-slider"] > div {
        background-color: #111;
    }
    /* Button styling */
    div[data-testid="stButton"] > button {
        background-color: #7B06A6;
        color: #fff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Sidebar Navigation ---
with st.sidebar:
    st.title("Prediction Apps", anchor=None)
    selected = option_menu(
        menu_title=None,
        options=["Home", "Relations & Correlations", "Prediction"],
        icons=["house", "bar-chart-line-fill", "calculator"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {"color": "white", "font-size": "16px", "text-align": "left", "margin": "0px", "margin-bottom": "15px"},
            "nav-link-selected": {"background-color": "#7B06A6", "font-size": "15px"},
        }
    )

# --- Pages ---
if selected == "Home":
    st.header("🏠 Welcome to Employee Salary Predictor")
    st.write("Use the sidebar to explore data or make salary predictions.")

elif selected == "Relations & Correlations":
    st.header("📈 Relations & Correlations")
    st.write("Explore feature correlations and scatter-matrix.")
    # Correlation heatmap
    corr = df.select_dtypes(include=np.number).corr()
    st.subheader("Correlation Heatmap")
    fig1 = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="Viridis")
    st.plotly_chart(fig1, use_container_width=True)

    # Scatter matrix
    st.subheader("Scatter Matrix")
    fig2 = px.scatter_matrix(
        df,
        dimensions=df.select_dtypes(include=np.number).columns.tolist(),
        color=df["Job Title"],
        title="Numeric Features Scatter Matrix"
    )
    fig2.update_traces(diagonal_visible=False)
    st.plotly_chart(fig2, use_container_width=True)

elif selected == "Prediction":
    st.header("💰 Employee Salary Prediction")
    st.write("Select a job title and years of experience to predict salary.")
    job = st.selectbox("Job Title", JOB_TITLES)
    exp = st.slider("Years of Experience", min_value=0, max_value=40, value=2)

    if st.button("Predict"):
        # Build input row
        row = {"const": 1, "Years of Experience": exp}
        for col in job_columns:
            row[col] = 1 if col == f"Job Title_{job}" else 0
        X_pred = pd.DataFrame([row])
        sqrt_pred = model.predict(X_pred)[0]
        salary_pred = sqrt_pred ** 2
        st.success(f"💵 Predicted Salary: ${salary_pred:,.0f}")
