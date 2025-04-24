import os
import sys
import pandas as pd
from train_lstm import train_lstm_model
from fx_lstm import predict_future_returns

# ---------- SETTINGS ----------
CURRENCY = sys.argv[1] if len(sys.argv) > 1 else "EUR"
CUTOFF_DATE = "2022-12-21"
PRED_END_DATE = "2025-02-01"
SAVE_DIR = f"./model/{CURRENCY}"
TIME_STEP = 10
EPOCHS = 100
BATCH_SIZE = 16

# ---------- LOAD DATA ----------
df_all = pd.read_csv("./data/returns_2014_2025_filtered.csv", parse_dates=["Date"])
df_all = df_all.set_index("Date")
df = df_all[[CURRENCY]].copy()

# ---------- TRAIN ----------
train_lstm_model(
    df=df,
    currency=CURRENCY,
    cutoff_date=CUTOFF_DATE,
    pred_end_date=PRED_END_DATE,
    time_step=TIME_STEP,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    save_dir=SAVE_DIR
)

# ---------- PREDICT ----------
predict_future_returns(
    df=df,
    currency=CURRENCY,
    final_model_path=f"{SAVE_DIR}/lstm_final_model.keras",
    scaler_path=f"{SAVE_DIR}/final_scaler.pkl",
    target_scaler_path=f"{SAVE_DIR}/target_scaler.pkl",
    save_path=f"{SAVE_DIR}/future_returns.csv"
)
