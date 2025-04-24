import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import os
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras import Input
from keras.losses import Huber


def compute_rsi(series, window=10):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def create_dataset(data, target, time_step):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i + time_step])
        y.append(target[i + time_step])
    return np.array(X), np.array(y)

def train_lstm_model(df, currency, cutoff_date, pred_end_date, time_step, epochs, batch_size, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    df = df[[currency]].copy()
    df["MA_5"] = df[currency].rolling(5).mean()
    df["MA_10"] = df[currency].rolling(10).mean()
    df["RSI"] = compute_rsi(df[currency]).rolling(3).mean()
    df["Momentum_5"] = df[currency] - df[currency].shift(5)
    df["Momentum_10"] = df[currency] - df[currency].shift(10)
    df["Volatility_5"] = df[currency].rolling(5).std()
    df["Volatility_10"] = df[currency].rolling(10).std()
    df["target"] = df[currency].shift(-1).rolling(5).mean()
    # Lagged target features
    for lag in [1, 2, 3, 5, 7]:
        df[f"lag_target_{lag}"] = df["target"].shift(lag)
    # Mean diff and Z-score
    mean_20 = df[currency].rolling(20).mean()
    std_20 = df[currency].rolling(20).std()
    df["MA_diff"] = df["MA_5"] - df["MA_10"]
    df["Z_Score"] = (df[currency] - mean_20) / std_20
    
    target_scaler = StandardScaler()
    df["target_scaled"] = target_scaler.fit_transform(df["target"].values.reshape(-1, 1)).flatten()
    df.dropna(inplace=True)

    features = [
        "MA_5", "MA_10", "RSI",
        "Momentum_5", "Momentum_10",
        "Volatility_5", "Volatility_10",
        "lag_target_1", "lag_target_2",
        "lag_target_3", "lag_target_5", "lag_target_7",
        "MA_diff", "Z_Score"
    ]

    X_full = df[features].values
    y_full = df["target_scaled"].values

    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_full)

    X_train_scaled = X_full_scaled[df.index < cutoff_date]
    y_train = y_full[df.index < cutoff_date]
    X_test_scaled = X_full_scaled[(df.index >= cutoff_date) & (df.index <= pred_end_date)]
    y_test = y_full[(df.index >= cutoff_date) & (df.index <= pred_end_date)]

    X_seq, y_seq = create_dataset(X_full_scaled, y_full, time_step)
    X_train = X_seq[df.index[time_step:] < cutoff_date]
    y_train = y_seq[df.index[time_step:] < cutoff_date]
    X_test = X_seq[(df.index[time_step:] >= cutoff_date) & (df.index[time_step:] <= pred_end_date)]
    y_test = y_seq[(df.index[time_step:] >= cutoff_date) & (df.index[time_step:] <= pred_end_date)]

    model = Sequential()
    model.add(Input(shape=(time_step, X_train.shape[2])))
    model.add(Conv1D(filters=32, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.4))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss=Huber(delta=1.0))


    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)
    history = model.fit(
        X_train_final, y_train_final,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )

    y_pred = model.predict(X_test)
    y_pred_inv = target_scaler.inverse_transform(y_pred)
    y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)

    print(f"{currency} — RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    model.save(f"{save_dir}/lstm_model.keras")
    joblib.dump(scaler, f"{save_dir}/scaler.pkl")

    n = len(y_test)
    test_dates = df[(df.index >= cutoff_date) & (df.index <= pred_end_date)].index
    date_index = test_dates[-n:]

    results_df = pd.DataFrame({
        "Date": date_index,
        "Currency": [currency] * n,
        "Actual": y_test_inv.flatten(),
        "Predicted": y_pred_inv.flatten(),
        "RMSE": [rmse] * n,
        "MAE": [mae] * n,
        "R2": [r2] * n
    })
    results_df.to_csv(f"{save_dir}/lstm_predictions.csv", index=False)

    plot_model(model, show_shapes=True, show_layer_names=True, to_file=f"{save_dir}/model.png")

    X_full_seq, y_full_seq = create_dataset(X_full_scaled, y_full, time_step)
    X_full_seq = X_full_seq.reshape(X_full_seq.shape[0], time_step, X_full_seq.shape[2])

    final_model = Sequential()
    final_model.add(Input(shape=(X_full_seq.shape[1], X_full_seq.shape[2])))
    final_model.add(LSTM(64, return_sequences=False))
    # final_model.add(LSTM(64, return_sequences=False, input_shape=(X_full_seq.shape[1], X_full_seq.shape[2])))
    final_model.add(Dropout(0.2))
    final_model.add(Dense(1))
    final_model.compile(optimizer='adam', loss='mse')
    final_model.fit(X_full_seq, y_full_seq, epochs=epochs, batch_size=batch_size, verbose=1)

    final_model.save(f"{save_dir}/lstm_final_model.keras")
    joblib.dump(scaler, f"{save_dir}/final_scaler.pkl")
    joblib.dump(target_scaler, f"{save_dir}/target_scaler.pkl")
