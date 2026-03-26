import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect("data/network_metrics.db")
df   = pd.read_sql("SELECT latency_ms FROM network_metrics", conn)
conn.close()

lat_min  = df["latency_ms"].min()
lat_max  = df["latency_ms"].max()
lat_mean = df["latency_ms"].mean()
lat_std  = df["latency_ms"].std()
lat_range = lat_max - lat_min

# Convert scaled MAE back to real milliseconds
mae_lstm  = 0.0312 * lat_range
mae_gru   = 0.0316 * lat_range
mae_rf    = 0.0370 * lat_range
mae_lr    = 0.0377 * lat_range
mae_xgb   = 0.0392 * lat_range

print("=" * 50)
print("  YOUR REAL RESULTS IN MILLISECONDS")
print("=" * 50)
print(f"\n  Latency Range : {lat_min:.1f} ms — {lat_max:.1f} ms")
print(f"  Latency Mean  : {lat_mean:.1f} ms")
print(f"  Latency Std   : {lat_std:.2f} ms")
print(f"  Latency Range : {lat_range:.1f} ms")

print("\n" + "=" * 50)
print("  Model         MAE (real ms)   Rank")
print("-" * 50)
print(f"  LSTM          {mae_lstm:.3f} ms        🥇")
print(f"  GRU           {mae_gru:.3f} ms        🥈")
print(f"  RandomForest  {mae_rf:.3f} ms        🥉")
print(f"  LinearReg     {mae_lr:.3f} ms        4️⃣")
print(f"  XGBoost       {mae_xgb:.3f} ms        5️⃣")
print("=" * 50)
print(f"\n  LSTM beats XGBoost by:")
improvement = ((mae_xgb - mae_lstm) / mae_xgb) * 100
print(f"  {improvement:.1f}% better MAE!")
print(f"\n  In plain English:")
print(f"  LSTM prediction is wrong by only {mae_lstm:.2f}ms on average")
print(f"  XGBoost is wrong by {mae_xgb:.2f}ms on average")
print("=" * 50)
