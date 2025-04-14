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

st.sidebar.header("⚙️ Step 1: Select a Currency")

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


# 2. Rolling Mean Visualization: window size chosen by user
st.subheader("📋 Take a brief look at the data")

start_date = datetime(2022, 12, 21)
end_date = data.index[-1]
st.markdown(f"""
You are currently viewing **{currency}** returns from **2022-12-21 to {end_date.date()}** with a rolling average overlay.  
Use the slider to **adjust the rolling window** — a longer window smooths short-term noise and reveals longer-term trends.
""")

window = st.slider("Choose rolling window size (in days)", min_value=5, max_value=120, value=30, step=5)
st.caption("📎 Tip: Use a shorter window (e.g. 10 days) for fast trends, or a longer one (90 days) to smooth noise.")

show_raw = st.checkbox("Show raw daily return line", value=True)

fig = plot_rolling_mean_chart(data, currency, window, show_raw=show_raw)
st.plotly_chart(fig, use_container_width=True)


# 3. LSTM Model Prediction
st.subheader("🔮 LSTM Model Prediction")

# --- Part 1: Evaluation Preview (Precomputed from historical test set) ---
st.markdown("### 📊 Historical Evaluation (2022-12-21 to 2025-02-01)")
st.markdown("These results come from a pre-trained LSTM model evaluated on real data. The chart shows how well our model captured FX return trends.")

# Load historical predictions
@st.cache_data
def load_lstm_eval(currency):
    return pd.read_csv(f"./model/{currency}/lstm_predictions.csv", parse_dates=["Date"])

eval_df = load_lstm_eval(currency)
st.line_chart(eval_df.set_index("Date")[['Actual', 'Predicted']])

if st.checkbox("Show evaluation metrics"):
    st.write(f"**RMSE**: {eval_df['RMSE'].iloc[0]:.4f}")
    st.write(f"**MAE**: {eval_df['MAE'].iloc[0]:.4f}")
    st.write(f"**R²**: {eval_df['R2'].iloc[0]:.4f}")

# --- Part 2: Predict Future Returns ---
st.markdown("### 📈 Predict Future Returns")
st.markdown("Use our final trained LSTM model to forecast upcoming FX return trends based on recent data.")

# Date range picker for future forecasting
future_days = st.slider("Forecast how many future days?", min_value=1, max_value=30, value=5, step=1)

# Load precomputed future returns
@st.cache_data
def load_future_forecasts(currency):
    return pd.read_csv(f"./model/{currency}/future_returns.csv", parse_dates=["Date"])

future_df = load_future_forecasts(currency)

# Slice by user-selected number of days
future_shown = future_df.head(future_days).set_index("Date")

# Display results
st.line_chart(future_shown)
st.dataframe(future_shown)