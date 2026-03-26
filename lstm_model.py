import numpy as np
import pandas as pd
import pickle
import os
import sqlite3
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from preprocessor import preprocess_pipeline, WINDOW_SIZE

MODEL_FILE   = "models/lstm_model.h5"
HISTORY_FILE = "models/lstm_history.pkl"

def build_lstm_model(input_shape):
    print("\n[LSTM] Building model...")
    model = Sequential([
        LSTM(128, input_shape=input_shape,
             return_sequences=True, name="lstm_1"),
        Dropout(0.2),
        LSTM(64, return_sequences=False, name="lstm_2"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mean_squared_error",
        metrics=["mae"]
    )
    model.summary()
    return model

def train_lstm(X, y):
    print("\n[LSTM] Splitting data 80/20...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    print(f"  Train samples: {X_train.shape[0]}")
    print(f"  Val samples  : {X_val.shape[0]}")
    model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
    os.makedirs("models", exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_FILE, save_best_only=True,
                        monitor="val_loss", verbose=0)
    ]
    print("\n[LSTM] Training started...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    with open(HISTORY_FILE, "wb") as f:
        pickle.dump(history.history, f)
    print(f"\n[LSTM] Model saved to {MODEL_FILE}")
    return model, X_val, y_val

def evaluate_model(model, X_val, y_val):
    print("\n[EVALUATION] Computing metrics...")
    y_pred = model.predict(X_val, verbose=0).flatten()
    mae  = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2   = r2_score(y_val, y_pred)
    print("\n" + "="*45)
    print("  LSTM Model Results")
    print("="*45)
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R2   : {r2:.4f}")
    print("="*45)
    y_mean = np.full_like(y_val, y_val.mean())
    base   = mean_absolute_error(y_val, y_mean)
    print(f"  Baseline MAE : {base:.4f}")
    print(f"  Improvement  : {((base-mae)/base*100):.1f}%")
    return {"mae": mae, "rmse": rmse, "r2": r2}

def predict_future_latency(steps_ahead=[1, 2, 3]):
    try:
        if not os.path.exists(MODEL_FILE):
            print("[PREDICT] Model file not found!")
            return None

        model = load_model(MODEL_FILE)

        # Load recent data
        conn = sqlite3.connect("data/network_metrics.db")
        df   = pd.read_sql(
            "SELECT * FROM network_metrics ORDER BY timestamp DESC LIMIT 50",
            conn
        )
        conn.close()

        if len(df) < 20:
            print("[PREDICT] Not enough data")
            return None

        df = df.sort_values("timestamp").reset_index(drop=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Fill missing
        df["latency_ms"]      = df["latency_ms"].fillna(0)
        df["packet_loss"]     = df["packet_loss"].fillna(0)
        df["jitter_ms"]       = df["jitter_ms"].fillna(0)
        df["throughput_mbps"] = df["throughput_mbps"].fillna(0)

        # Add same features as training — MUST MATCH 11 features!
        df["hour"]        = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_peak_hour"] = df["hour"].apply(
            lambda h: 1 if (8<=h<=10) or (18<=h<=21) else 0
        )
        for col in ["latency_ms","packet_loss","jitter_ms"]:
            df[f"{col}_roll5_mean"] = df[col].rolling(5, min_periods=1).mean()
            df[f"{col}_roll5_std"]  = df[col].rolling(5, min_periods=1).std().fillna(0)

        # Same 11 feature columns as training
        feature_cols = [
            "latency_ms", "packet_loss", "jitter_ms", "throughput_mbps",
            "hour", "day_of_week", "is_peak_hour",
            "latency_ms_roll5_mean", "latency_ms_roll5_std",
            "packet_loss_roll5_mean", "jitter_ms_roll5_mean"
        ]

        data   = df[feature_cols].values.astype(float)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data)

        sequence = scaled[-20:].copy()
        predictions = {}

        for step in steps_ahead:
            temp = sequence.copy()
            for _ in range(step):
                inp      = temp[-20:].reshape(1, 20, len(feature_cols))
                next_val = float(model.predict(inp, verbose=0)[0][0])
                new_row     = temp[-1].copy()
                new_row[0]  = next_val
                temp        = np.vstack([temp, new_row])

            dummy       = np.zeros((1, len(feature_cols)))
            dummy[0][0] = temp[-1][0]
            real_val    = scaler.inverse_transform(dummy)[0][0]
            pred_ms     = round(max(1.0, float(real_val)), 2)
            predictions[f"t+{step*30}s"] = pred_ms

        print(f"[PREDICT] Success: {predictions}")
        return predictions

    except Exception as e:
        print(f"[PREDICT] Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("="*50)
    print("  LSTM Latency Prediction — Training")
    print("="*50)
    result = preprocess_pipeline()
    if result[0] is None:
        print("[ERROR] Not enough data!")
        exit()
    df, X, y, feature_cols, rf_X, rf_y = result
    model, X_val, y_val = train_lstm(X, y)
    evaluate_model(model, X_val, y_val)
    print("\n[TEST] Future latency predictions:")
    preds = predict_future_latency(steps_ahead=[1, 2, 3])
    if preds:
        for step, val in preds.items():
            print(f"  {step} ahead : {val} ms")
    print("\n[DONE] LSTM complete!")
