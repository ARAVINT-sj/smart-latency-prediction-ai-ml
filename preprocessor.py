import pandas as pd
import numpy as np
import sqlite3
import pickle
import os
from sklearn.preprocessing import MinMaxScaler

DB_FILE        = "data/network_metrics.db"
PROCESSED_FILE = "data/processed_data.csv"
SCALER_FILE    = "models/scaler.pkl"
WINDOW_SIZE    = 20

def load_data():
    print("[1] Loading data...")
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM network_metrics ORDER BY timestamp", conn)
    conn.close()
    print(f"    Loaded {len(df)} rows")
    return df

def clean_data(df):
    print("[2] Cleaning data...")
    original_len = len(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["latency_ms"]      = df["latency_ms"].interpolate(method="linear")
    df["jitter_ms"]       = df["jitter_ms"].interpolate(method="linear")
    df["packet_loss"]     = df["packet_loss"].fillna(0)
    df["throughput_mbps"] = df["throughput_mbps"].fillna(0)
    df = df.dropna(subset=["latency_ms"])
    df = df[df["latency_ms"] <= 300]
    df = df[df["latency_ms"] >= 0]
    df = df[df["packet_loss"] <= 100]
    print(f"    Removed {original_len - len(df)} bad rows")
    print(f"    Remaining: {len(df)} rows")
    return df.reset_index(drop=True)

def add_time_features(df):
    print("[3] Adding time features...")
    df["hour"]         = df["timestamp"].dt.hour
    df["minute"]       = df["timestamp"].dt.minute
    df["day_of_week"]  = df["timestamp"].dt.dayofweek
    df["is_peak_hour"] = df["hour"].apply(
        lambda h: 1 if (8 <= h <= 10) or (18 <= h <= 21) else 0
    )
    return df

def add_rolling_features(df):
    print("[4] Adding rolling features...")
    for col in ["latency_ms", "packet_loss", "jitter_ms"]:
        df[f"{col}_roll5_mean"] = df[col].rolling(5, min_periods=1).mean()
        df[f"{col}_roll5_std"]  = df[col].rolling(5, min_periods=1).std().fillna(0)
    return df

def scale_features(df, feature_cols):
    print("[5] Scaling features...")
    os.makedirs("models", exist_ok=True)
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)
    print(f"    Scaler saved!")
    return df, scaler

def create_lstm_sequences(df, feature_cols, target_col="latency_ms", window=20):
    print(f"[6] Creating LSTM sequences...")
    X, y = [], []
    values  = df[feature_cols].values
    targets = df[target_col].values
    for i in range(window, len(df)):
        X.append(values[i - window:i])
        y.append(targets[i])
    X = np.array(X)
    y = np.array(y)
    print(f"    X shape: {X.shape} | y shape: {y.shape}")
    return X, y

def preprocess_pipeline():
    print("\n" + "="*50)
    print(" Preprocessing Pipeline")
    print("="*50)
    df = load_data()
    if len(df) < 50:
        print("[WARN] Less than 50 rows! Collect more data first.")
        return None, None, None, None, None, None
    df = clean_data(df)
    df = add_time_features(df)
    df = add_rolling_features(df)
    feature_cols = [
        "latency_ms", "packet_loss", "jitter_ms", "throughput_mbps",
        "hour", "day_of_week", "is_peak_hour",
        "latency_ms_roll5_mean", "latency_ms_roll5_std",
        "packet_loss_roll5_mean", "jitter_ms_roll5_mean"
    ]
    df, scaler = scale_features(df, feature_cols)
    df.to_csv(PROCESSED_FILE, index=False)
    print(f"[7] Processed data saved!")
    X, y = create_lstm_sequences(df, feature_cols, "latency_ms", WINDOW_SIZE)
    rf_features = df[feature_cols].values
    rf_labels   = df["label"].values
    print("\n[DONE] Preprocessing complete!")
    return df, X, y, feature_cols, rf_features, rf_labels

if __name__ == "__main__":
    result = preprocess_pipeline()
    if result[0] is not None:
        df, X, y, feature_cols, rf_X, rf_y = result
        print("\nSample processed data:")
        print(df[["timestamp","latency_ms","packet_loss","label"]].tail(5))