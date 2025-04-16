import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import os
import matplotlib.pyplot as plt
from keras.utils import plot_model

# ---------- SETTINGS ----------
CURRENCY = "JPY"  # change this to GBP or JPY to run separately
TIME_STEP = 10
EPOCHS = 50
SAVE_DIR = f"./model/{CURRENCY}"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- LOAD DATA ----------
df = pd.read_csv("./data/returns_2014_2025_filtered.csv", parse_dates=["Date"])
df = df.set_index("Date")

# ---------- FEATURE ENGINEERING ----------
def compute_rsi(series, window=10):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def predict_future_returns(df, currency, final_model_path, scaler_path, save_path, future_days=30, time_step=10):
    # Load model and scaler
    model = load_model(final_model_path)
    scaler = joblib.load(scaler_path)

    # Prepare input features
    features = ["MA", "RSI", currency]
    df = df[[currency]].copy()
    df["MA"] = df[currency].rolling(5).mean()
    df["RSI"] = compute_rsi(df[currency])
    df.dropna(inplace=True)

    # Use the last TIME_STEP rows to start
    recent_df = df[-time_step:].copy()
    print(recent_df)
    future_preds = []
    predicted_series = list(recent_df[currency])

    for _ in range(future_days):
        # Recalculate MA and RSI for recent predicted series
        temp_series = pd.Series(predicted_series)
        ma = temp_series.rolling(5).mean().iloc[-1]
        rsi = compute_rsi(temp_series).iloc[-1]

        # Prepare input row
        last_input = np.array([[ma, rsi, predicted_series[-1]]])
        scaled_input = scaler.transform(last_input)

        # Build input sequence
        recent_scaled = scaler.transform(recent_df[features])
        sequence = np.vstack([recent_scaled[1:], scaled_input])  # slide forward
        sequence = sequence.reshape(1, time_step, 3)

        # Predict next return
        pred_scaled = model.predict(sequence)[0, 0]
        template = np.zeros((1, 3))
        template[0, 0] = pred_scaled
        pred_real = scaler.inverse_transform(template)[0, 0]
        future_preds.append(pred_real)

        # Update context
        predicted_series.append(pred_real)
        new_row = pd.DataFrame([[ma, rsi, pred_real]], columns=features, index=[df.index[-1] + pd.Timedelta(days=len(future_preds))])
        recent_df = pd.concat([recent_df, new_row])[-time_step:]

    # Build result DataFrame
    start_date = df.index[-1] + pd.Timedelta(days=1)
    future_dates = pd.date_range(start=start_date, periods=future_days)
    future_df = pd.DataFrame({"Date": future_dates, "Predicted Return": future_preds})

    # Save to CSV
    future_df.to_csv(save_path, index=False)

    return future_df

# Predict future returns
final_model_path=f"./model/{CURRENCY}/lstm_final_model.keras"
scaler_path=f"./model/{CURRENCY}/final_scaler.pkl"
save_path=f"./model/{CURRENCY}/future_returns.csv"

future_df = predict_future_returns(
    df,
    currency=CURRENCY,
    final_model_path=final_model_path,
    scaler_path=scaler_path,
    save_path=save_path
)