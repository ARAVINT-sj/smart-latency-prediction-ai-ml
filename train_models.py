import os
import sqlite3
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

os.makedirs("models", exist_ok=True)
print("="*60)
print("  Smart Network Prediction — Model Training")
print("="*60)

# STEP 1 — LOAD DATA
print("\n[1/6] Loading data...")
DB_PATH = "network_data.db"
if not os.path.exists(DB_PATH):
    for name in ["network_metrics.db","data.db","network.db"]:
        if os.path.exists(name):
            DB_PATH = name
            break
if not os.path.exists(DB_PATH):
    print("ERROR: Database not found! Run collector.py first.")
    exit(1)

conn = sqlite3.connect(DB_PATH)
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
table_name = tables['name'].iloc[0]
df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
conn.close()
print(f"   Loaded {len(df)} rows from '{table_name}'")

if len(df) < 30:
    print("ERROR: Not enough data! Run collector.py longer.")
    exit(1)

# STEP 2 — CLEAN DATA
print("\n[2/6] Cleaning data...")
col_map = {}
for col in df.columns:
    cl = col.lower()
    if "latency" in cl or "rtt" in cl or "ping" in cl:
        col_map[col] = "latency"
    elif "jitter" in cl:
        col_map[col] = "jitter"
    elif "loss" in cl or "packet" in cl:
        col_map[col] = "packet_loss"
    elif "throughput" in cl or "speed" in cl:
        col_map[col] = "throughput"
    elif "time" in cl or "stamp" in cl:
        col_map[col] = "timestamp"
    elif "health" in cl or "status" in cl or "label" in cl:
        col_map[col] = "health_label"
df = df.rename(columns=col_map)

for c in ["latency","jitter","packet_loss","throughput"]:
    if c not in df.columns:
        df[c] = 0.0

df = df.dropna(subset=["latency"])
df = df[df["latency"] < 500]
df = df[df["latency"] > 0]
df = df.reset_index(drop=True)
print(f"   Clean rows: {len(df)}")

# STEP 3 — FEATURE ENGINEERING
print("\n[3/6] Engineering features...")
if "timestamp" in df.columns:
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_peak_hour"] = df["hour"].apply(lambda h: 1 if (8<=h<=10 or 18<=h<=22) else 0)
    except:
        df["hour"] = 12
        df["day_of_week"] = 0
        df["is_peak_hour"] = 0
else:
    df["hour"] = 12
    df["day_of_week"] = 0
    df["is_peak_hour"] = 0

df["rolling_avg_latency"] = df["latency"].rolling(5, min_periods=1).mean()
df["rolling_std_latency"] = df["latency"].rolling(5, min_periods=1).std().fillna(0)
df["rolling_avg_jitter"] = df["jitter"].rolling(5, min_periods=1).mean()
df["rolling_avg_loss"] = df["packet_loss"].rolling(5, min_periods=1).mean()

if "health_label" not in df.columns:
    def label(row):
        if row["latency"] > 100 or row["packet_loss"] > 10:
            return "Critical"
        elif row["latency"] > 50 or row["packet_loss"] > 2:
            return "Degraded"
        else:
            return "Normal"
    df["health_label"] = df.apply(label, axis=1)

print(f"   Health counts: {df['health_label'].value_counts().to_dict()}")

FEATURES = [
    "latency","jitter","packet_loss","throughput",
    "hour","day_of_week","is_peak_hour",
    "rolling_avg_latency","rolling_std_latency",
    "rolling_avg_jitter","rolling_avg_loss"
]

# STEP 4 — SCALE
print("\n[4/6] Scaling features...")
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[FEATURES] = scaler.fit_transform(df[FEATURES])
with open("models/scaler.pkl","wb") as f:
    pickle.dump(scaler, f)
print("   Scaler saved")

lat_idx = FEATURES.index("latency")
lat_min = scaler.data_min_[lat_idx]
lat_max = scaler.data_max_[lat_idx]

# STEP 5 — BUILD SEQUENCES
print("\n[5/6] Building sequences...")
WINDOW = 20
X_seq, y_seq = [], []
vals = df_scaled[FEATURES].values
lat  = df_scaled["latency"].values
for i in range(WINDOW, len(vals)):
    X_seq.append(vals[i-WINDOW:i])
    y_seq.append(lat[i])
X_seq = np.array(X_seq)
y_seq = np.array(y_seq)
split = int(len(X_seq)*0.8)
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]
X_flat_train = X_train[:,-1,:]
X_flat_test  = X_test[:,-1,:]
print(f"   Train: {len(X_train)}  Test: {len(X_test)}")

from sklearn.metrics import mean_absolute_error
results = {}

# STEP 6 — TRAIN MODELS
print("\n[6/6] Training models...")

# LSTM
print("\n Training LSTM...")
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    tf.random.set_seed(42)
    m = Sequential([
        LSTM(128, return_sequences=True, input_shape=(WINDOW,len(FEATURES))),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    m.compile(optimizer="adam", loss="mse")
    h = m.fit(X_train, y_train, validation_split=0.1,
              epochs=100, batch_size=32, verbose=1,
              callbacks=[EarlyStopping(patience=8, restore_best_weights=True)])
    pred = m.predict(X_test, verbose=0).flatten()
    mae  = mean_absolute_error(y_test, pred) * (lat_max - lat_min)
    m.save("models/lstm_model.h5")
    results["LSTM"] = f"{mae:.4f} ms"
    print(f"   LSTM MAE: {mae:.4f}ms  Saved!")
except Exception as e:
    print(f"   LSTM Error: {e}")

# GRU
print("\n Training GRU...")
try:
    from tensorflow.keras.layers import GRU
    g = Sequential([
        GRU(128, return_sequences=True, input_shape=(WINDOW,len(FEATURES))),
        Dropout(0.2),
        GRU(64),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    g.compile(optimizer="adam", loss="mse")
    g.fit(X_train, y_train, validation_split=0.1,
          epochs=100, batch_size=32, verbose=0,
          callbacks=[EarlyStopping(patience=8, restore_best_weights=True)])
    pred = g.predict(X_test, verbose=0).flatten()
    mae  = mean_absolute_error(y_test, pred) * (lat_max - lat_min)
    g.save("models/gru_model.h5")
    results["GRU"] = f"{mae:.4f} ms"
    print(f"   GRU MAE: {mae:.4f}ms  Saved!")
except Exception as e:
    print(f"   GRU Error: {e}")

# Random Forest
print("\n Training Random Forest...")
try:
    from sklearn.ensemble import RandomForestClassifier
    label_map = {"Normal":0,"Degraded":1,"Critical":2}
    y_clf = df["health_label"].map(label_map).values
    X_clf = df[FEATURES].values
    sp = int(len(X_clf)*0.8)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                class_weight="balanced", random_state=42)
    rf.fit(X_clf[:sp], y_clf[:sp])
    acc = rf.score(X_clf[sp:], y_clf[sp:]) * 100
    with open("models/rf_model.pkl","wb") as f: pickle.dump(rf, f)
    with open("models/label_map.pkl","wb") as f: pickle.dump(label_map, f)
    results["RandomForest"] = f"{acc:.1f}%"
    print(f"   RF Accuracy: {acc:.1f}%  Saved!")
except Exception as e:
    print(f"   RF Error: {e}")

# Linear Regression
print("\n Training Linear Regression...")
try:
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_flat_train, y_train)
    mae = mean_absolute_error(y_test, lr.predict(X_flat_test)) * (lat_max - lat_min)
    with open("models/lr_model.pkl","wb") as f: pickle.dump(lr, f)
    results["LinearRegression"] = f"{mae:.4f} ms"
    print(f"   LR MAE: {mae:.4f}ms  Saved!")
except Exception as e:
    print(f"   LR Error: {e}")

# XGBoost
print("\n Training XGBoost...")
try:
    from xgboost import XGBRegressor
    xgb = XGBRegressor(n_estimators=200, max_depth=6,
                       learning_rate=0.05, random_state=42, verbosity=0)
    xgb.fit(X_flat_train, y_train)
    mae = mean_absolute_error(y_test, xgb.predict(X_flat_test)) * (lat_max - lat_min)
    with open("models/xgb_model.pkl","wb") as f: pickle.dump(xgb, f)
    results["XGBoost"] = f"{mae:.4f} ms"
    print(f"   XGBoost MAE: {mae:.4f}ms  Saved!")
except Exception as e:
    print(f"   XGBoost Error: {e}")

# Save metadata
meta = {
    "features": FEATURES,
    "window_size": WINDOW,
    "n_samples": len(df),
    "results": results,
    "label_map": {"Normal":0,"Degraded":1,"Critical":2},
    "lat_min": float(df["latency"].min()),
    "lat_max": float(df["latency"].max()),
    "lat_avg": float(df["latency"].mean()),
}
with open("models/meta.pkl","wb") as f: pickle.dump(meta, f)

print("\n" + "="*60)
print("  TRAINING COMPLETE!")
print("="*60)
for k,v in results.items():
    print(f"  {k:20s} → {v}")
print("\n  Now run: python app.py")
print("="*60)
