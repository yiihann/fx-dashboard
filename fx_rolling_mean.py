import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Annotation dictionary
ANNOTATIONS = {
    'EUR': {
        '2023-07-20': "AI Boom",
        '2024-09-18': "Fed Cut",
        '2025-01-14': "USD Peak"
    },
    'JPY': {
        '2024-06-28': 'Yen Drop',
        '2023-07-20': "AI Boom",
        '2025-01-14': 'Yen Low',
    },
    'GBP': {
        '2023-07-01': 'GBP Peaks',
        '2025-01-14': 'Mkt Worry'
    }
}

def plot_rolling_mean_chart(data: pd.DataFrame, currency: str, window: int, show_raw: bool = True):
    """
    Generate a plotly figure showing daily returns and rolling average with annotations.
    """
    # Define start and end
    start_date = datetime(2022, 12, 21)
    end_date = data.index[-1]

    # Calculate rolling mean
    rolling_mean = data.rolling(window=window).mean()

    # Filter data
    filtered = data.loc[start_date:end_date]
    rolling_filtered = rolling_mean.loc[start_date:end_date]

    # Create figure
    fig = go.Figure()
    if show_raw:
        fig.add_trace(go.Scatter(x=filtered.index, y=filtered, name="Daily Return", line=dict(color="skyblue")))

    fig.add_trace(go.Scatter(x=rolling_filtered.index, y=rolling_filtered, name=f"Rolling Mean ({window}d)", line=dict(color="salmon")))

    # Annotate
    y_top = filtered.max()
    y_annot = y_top * 0.95
    for date, label in ANNOTATIONS[currency].items():
        date_obj = pd.to_datetime(date)
        if start_date <= date_obj <= end_date:
            fig.add_vline(x=date_obj, line=dict(color="gray", dash="dash"), opacity=0.5)
            fig.add_annotation(x=date_obj, y=y_annot, text=label, showarrow=False, textangle=90, yanchor="bottom", bgcolor='white')

    fig.update_layout(
        title=f"{currency} Return with {window}-Day Rolling Mean",
        xaxis_title="Date",
        yaxis_title="Return",
        legend_title="Series",
        hovermode="x unified"
    )

    return fig
