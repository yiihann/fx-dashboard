import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from fx_rolling_mean import plot_rolling_mean_chart

st.set_page_config(page_title="FX Prediction Dashboard", layout="centered")

st.title("📈 FX Return Prediction Tool")
st.markdown("""
Welcome! This tool lets you explore historical FX return trends and LSTM model predictions
for EUR, GBP, and JPY. You can use the sidebar to select a currency, choose a time period, and view results.
""")

st.sidebar.title("🛠️ FX Prediction Settings")

section = st.sidebar.radio(
    "🧭 Navigation",
    ["Rolling Mean", "Model Evaluation", "Forecast", "Strategy Simulator"]
)

st.sidebar.header("⚙️ Select a Currency")


# Load your fixed return dataset
@st.cache_data
def load_data():
    df = pd.read_csv("./data/returns_2014_2025_filtered.csv", parse_dates=["Date"])
    return df

df = load_data().dropna()
df = df.set_index("Date")

# 1. Let user select currency
available_currencies = list(df.columns)
currency = st.sidebar.selectbox("Choose a currency to predict", available_currencies)
data = df[currency]
st.success(f"Using historical return data for **{currency}** from 2014 to 2025.")

# Load historical predictions
@st.cache_data
def load_lstm_eval(currency):
    return pd.read_csv(f"./model/{currency}/lstm_predictions.csv", parse_dates=["Date"])

eval_df = load_lstm_eval(currency)

# Load precomputed future returns
@st.cache_data
def load_future_forecasts(currency):
    return pd.read_csv(f"./model/{currency}/future_returns.csv", parse_dates=["Date"])

future_df = load_future_forecasts(currency)

if section == "Rolling Mean":
    st.subheader("📋 Take a brief look at the data")

    start_date = datetime(2022, 12, 21)
    end_date = data.index[-1]
    st.markdown(f"""
    You are currently viewing **{currency}** returns from **2022-12-21 to {end_date.date()}** with a rolling average overlay.  
    Use the slider to **adjust the rolling window** — a longer window smooths short-term noise and reveals longer-term trends.
    """)

    window = st.slider("Choose rolling window size (in days)", min_value=5, max_value=120, value=30, step=5)
    st.caption("📎 Tip: Use a shorter window (e.g. 10 days) for fast trends, or a longer one (90 days) to smooth noise.")

    show_raw = st.checkbox("📋 Show raw daily return line", value=True)

    fig = plot_rolling_mean_chart(data, currency, window, show_raw=show_raw)
    st.plotly_chart(fig, use_container_width=True)

if section == "Model Evaluation":
    st.subheader("🔮 LSTM Model Prediction")

    # --- Part 1: Evaluation Preview (Precomputed from historical test set) ---
    st.markdown("### 📊 Historical Evaluation (2022-12-21 to 2025-02-01)")
    st.markdown("These results come from a pre-trained LSTM model evaluated on real data. The chart shows how well our model captured 5-day average return trends.")

    fig_eval = go.Figure()
    fig_eval.add_trace(go.Scatter(x=eval_df["Date"], y=eval_df["Actual"], mode='lines', name='Actual', line=dict(color='skyblue')))
    fig_eval.add_trace(go.Scatter(x=eval_df["Date"], y=eval_df["Predicted"], mode='lines', name='Predicted', line=dict(color='salmon')))
    fig_eval.add_trace(go.Scatter(
        x=eval_df["Date"],
        y=eval_df["Predicted"] + eval_df["RMSE"],
        mode='lines',
        line=dict(width=0),
        name='+1 RMSE',
        showlegend=False
    ))
    fig_eval.add_trace(go.Scatter(
        x=eval_df["Date"],
        y=eval_df["Predicted"] - eval_df["RMSE"],
        fill='tonexty',
        mode='lines',
        line=dict(width=0),
        name='-1 RMSE',
        fillcolor='rgba(250, 128, 114, 0.2)',
        showlegend=True
    ))
    fig_eval.update_layout(title="LSTM Historical Evaluation with Confidence Band", xaxis_title="Date", yaxis_title="5-Day Avg Return")
    st.plotly_chart(fig_eval, use_container_width=True)
    if st.checkbox("📋 Show evaluation metrics"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("RMSE", f"{eval_df['RMSE'].iloc[0]:.4f}")
        with col2:
            st.metric("MAE", f"{eval_df['MAE'].iloc[0]:.4f}")
        with col3:
            st.metric("R²", f"{eval_df['R2'].iloc[0]:.4f}")

    # ----------- Residual Plot ----------- #
    residuals = eval_df["Actual"] - eval_df["Predicted"]
    fig_resid = go.Figure()
    fig_resid.add_trace(go.Scatter(
        x=eval_df["Date"],
        y=residuals,
        mode='lines+markers',
        name='Residuals',
        line=dict(color='indianred'),
        marker=dict(color='indianred')
    ))
    fig_resid.update_layout(
        title="Prediction Residuals (Actual - Predicted)",
        xaxis_title="Date",
        yaxis_title="Residual",
        hovermode="x unified",
        plot_bgcolor='white',
        margin=dict(l=40, r=40, t=60, b=40)
    )
    fig_resid.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig_resid.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    st.plotly_chart(fig_resid, use_container_width=True)
    st.caption("📉 This chart displays the prediction errors (residuals) over time. Ideally, residuals should fluctuate randomly around zero — suggesting unbiased forecasts.")

if section == "Forecast" or section == "Strategy Simulator":
    st.sidebar.header("⚙️ Select a Prediction Horizon")
    future_days = st.sidebar.slider("📆 Forecast Horizon (Days)", min_value=1, max_value=30, value=20, step=1)

    # Slice by user-selected number of days
    future_shown = future_df.head(future_days).set_index("Date")
    future_shown.index = future_shown.index.strftime("%Y-%m-%d")
    future_shown = future_shown.reset_index().rename(columns={"index": "Date"})


if section == "Forecast":
    st.markdown("### 📈 Predict Future 5-Day Average Returns")
    st.markdown("These predictions reflect our model's forecast of smoothed short-term return trends.")

    # # Date range picker for future forecasting
    # future_days = st.slider("Forecast how many future days?", min_value=1, max_value=30, value=20, step=1)

    # Display results
    fig_future = go.Figure()

    fig_future.add_trace(go.Scatter(x=future_shown.index, y=future_shown["Predicted Return"], mode='lines', name='Predicted', line=dict(color='salmon')))

    # Estimate confidence band using std dev if available or fallback to RMSE from historical eval
    future_rmse = eval_df["RMSE"].iloc[0] if "RMSE" in eval_df else 0.001  # default fallback
    fig_future.add_trace(go.Scatter(
        x=future_shown.index,
        y=future_shown["Predicted Return"] + future_rmse,
        mode='lines',
        line=dict(width=0),
        name='+1 RMSE',
        showlegend=False
    ))
    fig_future.add_trace(go.Scatter(
        x=future_shown.index,
        y=future_shown["Predicted Return"] - future_rmse,
        fill='tonexty',
        mode='lines',
        line=dict(width=0),
        name='Confidence Interval',
        fillcolor='rgba(250, 128, 114, 0.2)',
        showlegend=True
    ))

    fig_future.update_layout(title="Forecasted 5-Day Returns with Confidence Band", xaxis_title="Date", yaxis_title="Return")
    st.plotly_chart(fig_future, use_container_width=True)

    if st.checkbox("📋 Show Forecast Table", value=True):
        st.caption("Forecasted 5-day average return for the selected currency.")
        st.table(future_shown.style.format({"Predicted Return": "{:.4f}"}))
        
if section == "Strategy Simulator":
    st.markdown("### 📊 Strategy Simulator")
    st.markdown("This simulator tests a simple trading rule: **If predicted return > threshold, go long. Otherwise, hold.**")

    buy_threshold = st.slider("Set prediction threshold for triggering a BUY", min_value=0.000, max_value=0.005, value=0.001, step=0.0001, format="%0.4f")
    short_threshold = st.slider("Set prediction threshold for triggering a SHORT", min_value=-0.005, max_value=0.000, value=-0.001, step=0.0001, format="%0.4f")

    simulated_return = future_shown["Predicted Return"].copy()
    signals_long = simulated_return > buy_threshold
    signals_short = simulated_return < short_threshold

    strategy_returns = pd.Series(0.0, index=simulated_return.index)
    strategy_returns[signals_long] = simulated_return[signals_long]
    strategy_returns[signals_short] = -simulated_return[signals_short]

    if strategy_returns.sum() == 0:
        st.warning("⚠️ No long or short signals triggered. Try adjusting the thresholds.")

    cumulative_returns = (1 + strategy_returns).cumprod()

    fig_strategy = go.Figure()
    fig_strategy.add_trace(go.Scatter(x=future_shown.index, y=cumulative_returns, mode='lines', name='Cumulative Return', line=dict(color='green')))
    fig_strategy.update_layout(title="Simulated Cumulative Return Based on Strategy", xaxis_title="Date", yaxis_title="Cumulative Return")
    st.plotly_chart(fig_strategy, use_container_width=True)
    st.caption("Note: This is a toy simulation using predicted return as proxy for actual return.")
