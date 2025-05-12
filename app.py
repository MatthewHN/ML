# Importing ToolKits
import relations
import prediction

import pandas as pd
import pickle

import streamlit as st
from streamlit_option_menu import option_menu
import warnings


def run():
    # Page config
    st.set_page_config(
        page_title="Salary Prediction",
        page_icon="📊",
        layout="wide"
    )

    # Suppress future warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # --- Data & Model Loaders ---
    @st.cache_data
    def load_data(path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    @st.cache_data
    def load_model(path: str):
        with open(path, "rb") as f:
            saved = pickle.load(f)
        return saved["model"], saved["job_columns"]

    # Load CSV and trained model
    df = load_data("Salary_Data.csv")
    model, job_columns = load_model("model.pkl")

    # CSS Styles (unchanged)
    st.markdown(
        """
        <style>
             .main { text-align: center; }
             .st-emotion-cache-1ibsh2c { padding-left: 3rem; padding-right: 3rem; }
             .st-emotion-cache-16txtl3 h1 { font: bold 29px arial; text-align: center; margin-bottom: 15px; }
             div[data-testid=stSidebarContent] { background-color: #111; border-right: 4px solid #222; padding: 8px!important; }
             div.block-containers { padding-top: 0.5rem; }
             .st-emotion-cache-z5fcl4 { padding: 1rem 2.2rem 1rem 1.1rem; overflow-x: hidden; }
             .st-emotion-cache-16txtl3 { padding: 2.7rem 0.6rem; }
             .plot-container.plotly { border: 1px solid #333; border-radius: 6px; }
             div.st-emotion-cache-1r6slb0 span.st-emotion-cache-10trblm { font: bold 24px tahoma; }
             div [data-testid=stImage] { text-align: center; display: block; margin: auto; width: 100%; }
             div[data-baseweb=select] > div { cursor: pointer; background-color: #111; border: 2px solid purple; }
             div[data-baseweb=base-input] { background-color: #111; border: 4px solid #444; border-radius: 5px; padding: 5px; }
             div[data-testid=stFormSubmitButton] > button { width: 100%; background-color: #111; border: 2px solid violet; padding: 18px; border-radius: 30px; opacity: 0.8; }
             div[data-testid=stFormSubmitButton] p { font-weight: bold; font-size: 20px; }
             div[data-testid=stFormSubmitButton] > button:hover { opacity: 1; border: 2px solid violet; color: #fff; }
             /* Hide default Streamlit menu/footer */
             #MainMenu, footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Sidebar styling
    side_bar_options_style = {
        "container": {"padding": "0!important", "background-color": 'transparent'},
        "icon": {"color": "white", "font-size": "18px"},
        "nav-link": {"color": "white", "font-size": "16px", "text-align": "left", "margin": "0 0 15px 0"},
        "nav-link-selected": {"background-color": "#7B06A6", "font-size": "15px"},
    }

    # Main containers
    header = st.container()
    content = st.container()

    # Sidebar navigation
    with st.sidebar:
        st.title("Machine Learning project by Matthew Horne")
        page = option_menu(
            menu_title=None,
            options=['Home', 'Relations & Correlations', 'Prediction'],
            icons=['diagram-3-fill', 'bar-chart-line-fill', 'graph-up-arrow'],
            menu_icon="cast",
            default_index=0,
            styles=side_bar_options_style
        )

    # Home Page
    if page == 'Home':
        with header:
            st.header('Employee Salary Prediction Data 📈💰')
        with content:
            st.dataframe(df, use_container_width=True)

    # Relations & Correlations
    elif page == 'Relations & Correlations':
        with header:
            st.header('Correlations Between Data 📉🚀')
        with content:
            st.plotly_chart(relations.create_heat_map(df), use_container_width=True)
            st.plotly_chart(relations.create_scatter_matrix(df), use_container_width=True)
            st.write("***")
            col1, col2 = st.columns(2)
            with col1:
                first_feature = st.selectbox(
                    "First Feature", df.select_dtypes(include="number").columns.tolist(), index=0
                )
            with col2:
                second_features = df.select_dtypes(include="number").columns.drop(first_feature)
                second_feature = st.selectbox("Second Feature", second_features, index=0)
            st.plotly_chart(relations.create_relation_scatter(df, first_feature, second_feature), use_container_width=True)

    # Prediction Page
    else:
        with header:
            st.header('Prediction Model 💰🔥')
        with content:
            st.markdown("Select a job title & years of experience to predict salary.")
            job_titles = [col.replace("Job Title_", "") for col in job_columns]
            job = st.selectbox("Job Title", job_titles)
            exp = st.slider("Years of Experience", min_value=0, max_value=40, value=2)

            if st.button('Predict', use_container_width=True):
                # Build input vector
                row = {"const": 1, "Years of Experience": exp}
                for col in job_columns:
                    row[col] = 1 if col == f"Job Title_{job}" else 0
                X_pred = pd.DataFrame([row])
                prediction_value = model.predict(X_pred)[0]
                st.success(f"💵 Predicted Salary: ${prediction_value:,.0f}")

# Run the app
run()
