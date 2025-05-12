import pandas as pd
import plotly.express as px
import streamlit as st

def create_comparison_df(y_true, y_pred):
    """
    Build a DataFrame with Actual, Predicted, and Residuals.
    """
    # y_true may come as a DataFrame or array
    if isinstance(y_true, pd.DataFrame):
        actual = y_true.iloc[:, 0].values
    else:
        actual = pd.Series(y_true).squeeze().values
    pred = pd.Series(y_pred).squeeze().values

    df = pd.DataFrame({
        "Actual": actual,
        "Predicted": pred
    })
    df["Residuals"] = df["Actual"] - df["Predicted"]
    return df

def create_residules_scatter(df):
    """
    Plot Predicted vs Residuals with a trendline.
    """
    fig = px.scatter(
        df,
        x="Predicted",
        y="Residuals",
        title="Residuals vs Predicted",
        trendline="ols",
        labels={"Predicted":"Predicted Salary", "Residuals":"Residual"}
    )
    fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
    return fig

def creat_matrix_score_cards(icon_path, title, value, is_percent):
    """
    In a Streamlit column, render an icon + metric.
    """
    st.image(icon_path, width=50)
    st.subheader(title)
    if is_percent:
        st.subheader(f"{value:.2f}%")
    else:
        st.subheader(f"{value:,.2f}")
