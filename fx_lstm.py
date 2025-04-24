import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys

def compute_rsi(series, window=10):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def predict_future_returns(df, currency, final_model_path, scaler_path, target_scaler_path, save_path, future_days=30, time_step=20):
    model = load_model(final_model_path)
    scaler = joblib.load(scaler_path)
    target_scaler = joblib.load(target_scaler_path)

    df = df[[currency]].copy()
    df["MA_5"] = df[currency].rolling(5).mean()
    df["MA_10"] = df[currency].rolling(10).mean()
    df["RSI"] = compute_rsi(df[currency]).rolling(3).mean()
    df["Momentum_5"] = df[currency] - df[currency].shift(5)
    df["Momentum_10"] = df[currency] - df[currency].shift(10)
    df["Volatility_5"] = df[currency].rolling(5).std()
    df["Volatility_10"] = df[currency].rolling(10).std()
    df["target"] = df[currency].shift(-1).rolling(5).mean()
    df["lag_target_1"] = df["target"].shift(1)
    df["lag_target_2"] = df["target"].shift(2)
    df["lag_target_3"] = df["target"].shift(3)
    df["lag_target_5"] = df["target"].shift(5)
    df["lag_target_7"] = df["target"].shift(7)
    df["MA_diff"] = df["MA_5"] - df["MA_10"]
    df["Z_Score"] = (df[currency] - df[currency].rolling(20).mean()) / df[currency].rolling(20).std()
    df.dropna(inplace=True)

    features = [
        "MA_5", "MA_10", "RSI",
        "Momentum_5", "Momentum_10",
        "Volatility_5", "Volatility_10",
        "lag_target_1", "lag_target_2", "lag_target_3", "lag_target_5", "lag_target_7",
        "MA_diff", "Z_Score"
    ]

    # Scale the entire feature set using DataFrame
    df_scaled = pd.DataFrame(
        scaler.transform(df[features]),
        columns=features,
        index=df.index
    )

    recent_df = df_scaled[-time_step:].copy()
    original_df = df[-time_step:].copy()  # to fetch most recent original values
    future_preds = []
    predicted_series = list(df["target"].iloc[-time_step:].values)

    for _ in range(future_days):
        temp_series = pd.Series(predicted_series)
        lag_1 = temp_series.shift(1).iloc[-1]
        lag_2 = temp_series.shift(2).iloc[-1]
        lag_3 = temp_series.shift(3).iloc[-1]
        lag_5 = temp_series.shift(5).iloc[-1]
        lag_7 = temp_series.shift(7).iloc[-1]
        
        last_original = original_df.iloc[-1][["MA_5", "MA_10", "RSI", "Momentum_5", "Momentum_10", "Volatility_5", "Volatility_10", "MA_diff", "Z_Score"]].tolist()
        temp_input = last_original + [lag_1, lag_2, lag_3, lag_5, lag_7]

        input_df = pd.DataFrame([temp_input], columns=features)
        input_scaled = pd.DataFrame(scaler.transform(input_df), columns=features)

        sequence = pd.concat([recent_df.iloc[1:], input_scaled], ignore_index=True)
        sequence_array = sequence.to_numpy().reshape(1, time_step, len(features))

        pred_scaled = model.predict(sequence_array)[0, 0]
        pred_real = target_scaler.inverse_transform([[pred_scaled]])[0, 0]
        future_preds.append(pred_real)

        new_row = input_scaled.copy()
        new_row.index = [df.index[-1] + pd.Timedelta(days=len(future_preds))]
        recent_df = pd.concat([recent_df, new_row])[-time_step:]
        predicted_series.append(pred_scaled)

    start_date = df.index[-1] + pd.Timedelta(days=1)
    future_dates = pd.date_range(start=start_date, periods=future_days)
    future_df = pd.DataFrame({"Date": future_dates, "Predicted Return": future_preds})

    future_df.to_csv(save_path, index=False)
    return future_df

if __name__ == "__main__":
    CURRENCY = sys.argv[1] if len(sys.argv) > 1 else "EUR"
    CUTOFF_DATE = "2022-12-21"
    PRED_END_DATE = "2025-02-01"
    SAVE_DIR = f"./model/{CURRENCY}"
    TIME_STEP = 20
    EPOCHS = 100
    BATCH_SIZE = 32
    
    df_all = pd.read_csv("./data/returns_2014_2025_filtered.csv", parse_dates=["Date"])
    df_all = df_all.set_index("Date")
    df = df_all[[CURRENCY]].copy()
    
    predict_future_returns(
        df=df,
        currency=CURRENCY,
        final_model_path=f"{SAVE_DIR}/lstm_final_model.keras",
        scaler_path=f"{SAVE_DIR}/final_scaler.pkl",
        target_scaler_path=f"{SAVE_DIR}/target_scaler.pkl",
        save_path=f"{SAVE_DIR}/future_returns.csv"
    )