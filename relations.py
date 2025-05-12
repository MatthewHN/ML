import numpy as np
import plotly.express as px

def create_heat_map(df):
    """
    Returns a Plotly heatmap of the numeric-feature correlations.
    """
    corr = df.select_dtypes(include=np.number).corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Viridis",
        title="Feature Correlation Heatmap"
    )
    fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
    return fig

def create_scatter_matrix(df):
    """
    Returns a Plotly scatter-matrix of all numeric features,
    colored by the Job Title.
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    fig = px.scatter_matrix(
        df,
        dimensions=num_cols,
        color=df["Job Title"],
        title="Numeric Features Scatter Matrix"
    )
    fig.update_traces(diagonal_visible=False)
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig

def create_relation_scatter(df, feat1, feat2):
    """
    Returns a Plotly scatter of feat1 vs feat2,
    colored by Job Title.
    """
    fig = px.scatter(
        df,
        x=feat1,
        y=feat2,
        color=df["Job Title"],
        title=f"{feat1} vs {feat2}"
    )
    fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
    return fig
