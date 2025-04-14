import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import os
import matplotlib.pyplot as plt
from keras.utils import plot_model

# ---------- SETTINGS ----------
CURRENCY = "GBP"  # change this to GBP or JPY to run separately
CUTOFF_DATE = "2022-12-21"
PRED_END_DATE = "2025-02-01"
TIME_STEP = 10
EPOCHS = 50
SAVE_DIR = f"./model/{CURRENCY}"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- LOAD DATA ----------
df = pd.read_csv("./data/returns_2014_2025_filtered.csv", parse_dates=["Date"])
df = df.set_index("Date")
df = df[[CURRENCY]].copy()

# ---------- FEATURE ENGINEERING ----------
def compute_rsi(series, window=10):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df["MA"] = df[CURRENCY].rolling(5).mean()
df["RSI"] = compute_rsi(df[CURRENCY])
df.dropna(inplace=True)

# ---------- TRAIN/TEST SPLIT FOR PERFORMANCE VISUALIZATION ----------
train = df[df.index < CUTOFF_DATE]
test = df[(df.index >= CUTOFF_DATE) & (df.index <= PRED_END_DATE)]

# ---------- SCALE FEATURES ----------
features = ["MA", "RSI", CURRENCY]
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train[features])
test_scaled = scaler.transform(test[features])

# ---------- BUILD SEQUENCES ----------
def create_dataset(data, time_step):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:i+time_step])
        y.append(data[i+time_step, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_dataset(train_scaled, TIME_STEP)
X_test, y_test = create_dataset(test_scaled, TIME_STEP)

# ---------- RESHAPE ----------
X_train = X_train.reshape(X_train.shape[0], TIME_STEP, X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], TIME_STEP, X_test.shape[2])

# ---------- BUILD MODEL ----------
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))

# ---------- Split training into train and validation ----------
from sklearn.model_selection import train_test_split

X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train, test_size=0.2, shuffle=False  # no shuffle for time series
)

# ---------- COMPILE AND TRAIN ----------
model.compile(optimizer='adam', loss='mse')
history = model.fit(
    X_train_final, y_train_final,
    epochs=EPOCHS,
    batch_size=16,
    verbose=1,
    validation_data=(X_val, y_val)
)

# ---------- PLOT LOSS HISTORY ----------
plt.figure(figsize=(8, 4),dpi=200)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Loss History_{CURRENCY}')
plt.legend()
plt.savefig(f"{SAVE_DIR}/loss_history.png")
# plt.show()

# ---------- SAVE MODEL STRUCTURE PLOT ----------
plot_model(model, show_shapes=True, show_layer_names=True, to_file=f'{SAVE_DIR}/model.png')

# ---------- PREDICT ----------
pred_scaled = model.predict(X_test)

# ---------- INVERSE TRANSFORM ----------
pred_template = np.zeros((len(pred_scaled), 3))
pred_template[:, 0] = pred_scaled.flatten()
predicted_returns = scaler.inverse_transform(pred_template)[:, 0]

actual_template = np.zeros((len(y_test), 3))
actual_template[:, 0] = y_test
actual_returns = scaler.inverse_transform(actual_template)[:, 0]

# ---------- EVALUATION ----------
rmse = np.sqrt(mean_squared_error(actual_returns, predicted_returns))
mae = mean_absolute_error(actual_returns, predicted_returns)
r2 = r2_score(actual_returns, predicted_returns)

print(f"{CURRENCY} — RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

# ---------- SAVE MODEL & RESULTS ----------
model.save(f"{SAVE_DIR}/lstm_model.keras")
joblib.dump(scaler, f"{SAVE_DIR}/scaler.pkl")

results_df = pd.DataFrame({
    "Date": test.index[TIME_STEP+1:],
    "Currency": CURRENCY,
    "Actual": actual_returns,
    "Predicted": predicted_returns,
    "RMSE": rmse,
    "MAE": mae,
    "R2": r2
})
results_df.to_csv(f"{SAVE_DIR}/lstm_predictions.csv", index=False)

# ---------- FINAL MODEL TRAINING FOR FUTURE PREDICTION ----------
# Retrain on all available data
full_scaled = scaler.fit_transform(df[features])
X_full, y_full = create_dataset(full_scaled, TIME_STEP)
X_full = X_full.reshape(X_full.shape[0], TIME_STEP, X_full.shape[2])

final_model = Sequential()
final_model.add(LSTM(50, return_sequences=True, input_shape=(X_full.shape[1], X_full.shape[2])))
final_model.add(Dropout(0.2))
final_model.add(LSTM(50))
final_model.add(Dropout(0.2))
final_model.add(Dense(1))
final_model.compile(optimizer='adam', loss='mse')
final_model.fit(X_full, y_full, epochs=EPOCHS, batch_size=16, verbose=1)

# Save final model and scaler
final_model.save(f"{SAVE_DIR}/lstm_final_model.keras")
joblib.dump(scaler, f"{SAVE_DIR}/final_scaler.pkl")
