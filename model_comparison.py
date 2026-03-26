"""
============================================================
MODEL COMPARISON ENGINE
============================================================
Trains and compares 5 ML models:
  1. LSTM      (Deep Learning — sequence based)
  2. GRU       (Deep Learning — faster than LSTM)
  3. XGBoost   (Gradient Boosting — tree based)
  4. Linear Regression (Simple baseline)
  5. Random Forest Regressor (Ensemble)
Generates 6 comparison graphs automatically.
============================================================
"""

import numpy as np
import pandas as pd
import sqlite3
import os
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore")

DB_FILE     = "data/network_metrics.db"
RESULTS_DIR = "data/comparison_results"
WINDOW_SIZE = 20
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

COLORS = {
    "LSTM"             : "#58a6ff",
    "GRU"              : "#3fb950",
    "XGBoost"          : "#f78166",
    "LinearRegression" : "#d29922",
    "RandomForest"     : "#bc8cff"
}

def load_and_prepare_data():
    print("\n[DATA] Loading from database...")
    conn = sqlite3.connect(DB_FILE)
    df   = pd.read_sql(
        "SELECT * FROM network_metrics ORDER BY timestamp", conn
    )
    conn.close()
    print(f"[DATA] Loaded {len(df)} rows")

    df["timestamp"]       = pd.to_datetime(df["timestamp"])
    df["latency_ms"]      = df["latency_ms"].interpolate().fillna(0)
    df["packet_loss"]     = df["packet_loss"].fillna(0)
    df["jitter_ms"]       = df["jitter_ms"].fillna(0)
    df["throughput_mbps"] = df["throughput_mbps"].fillna(0)
    df = df[df["latency_ms"] <= 300]
    df = df[df["latency_ms"] >= 0]

    df["hour"]          = df["timestamp"].dt.hour
    df["day_of_week"]   = df["timestamp"].dt.dayofweek
    df["is_peak"]       = df["hour"].apply(
        lambda h: 1 if (8<=h<=10) or (18<=h<=21) else 0
    )
    df["latency_roll5"] = df["latency_ms"].rolling(5, min_periods=1).mean()
    df["latency_std5"]  = df["latency_ms"].rolling(5, min_periods=1).std().fillna(0)
    df["loss_roll5"]    = df["packet_loss"].rolling(5, min_periods=1).mean()
    df["jitter_roll5"]  = df["jitter_ms"].rolling(5, min_periods=1).mean()

    feature_cols = [
        "latency_ms", "packet_loss", "jitter_ms", "throughput_mbps",
        "hour", "day_of_week", "is_peak",
        "latency_roll5", "latency_std5", "loss_roll5", "jitter_roll5"
    ]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[feature_cols].values)

    with open("models/comparison_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # ── LSTM/GRU sequences — predict 5 steps AHEAD ──────
    X_seq, y_seq = [], []
    for i in range(WINDOW_SIZE, len(scaled)):
        X_seq.append(scaled[i-WINDOW_SIZE:i])
        if i + 5 < len(scaled):
            y_seq.append(scaled[i+5][0])   # 5 steps ahead
        else:
            y_seq.append(scaled[i][0])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # ── Tabular — predict 5 steps AHEAD ─────────────────
    X_tab = scaled[WINDOW_SIZE:]
    y_tab = np.array([
        scaled[i+5][0] if i+5 < len(scaled) else scaled[i][0]
        for i in range(WINDOW_SIZE, len(scaled))
    ])

    print(f"[DATA] Sequence shape : X={X_seq.shape}, y={y_seq.shape}")
    print(f"[DATA] Tabular shape  : X={X_tab.shape}, y={y_tab.shape}")

    return X_seq, y_seq, X_tab, y_tab, scaler, feature_cols


def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ], name="LSTM")
    model.compile(optimizer=Adam(0.001), loss="mse", metrics=["mae"])
    return model


def build_gru(input_shape):
    model = Sequential([
        GRU(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        GRU(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ], name="GRU")
    model.compile(optimizer=Adam(0.001), loss="mse", metrics=["mae"])
    return model


def compute_metrics(y_true, y_pred, name):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {
        "model": name,
        "mae"  : round(mae,  4),
        "rmse" : round(rmse, 4),
        "r2"   : round(r2,   4),
        "mse"  : round(mean_squared_error(y_true, y_pred), 4)
    }


def train_all_models(X_seq, y_seq, X_tab, y_tab):
    split = int(len(y_seq) * 0.8)

    X_seq_train, X_seq_test = X_seq[:split], X_seq[split:]
    y_seq_train, y_seq_test = y_seq[:split], y_seq[split:]
    X_tab_train, X_tab_test = X_tab[:split], X_tab[split:]
    y_tab_train, y_tab_test = y_tab[:split], y_tab[split:]

    results   = {}
    histories = {}

    early_stop = EarlyStopping(
        monitor="val_loss", patience=8,
        restore_best_weights=True, verbose=0
    )

    # ── MODEL 1: LSTM ─────────────────────────────────────
    print("\n" + "="*50)
    print("[1/5] Training LSTM...")
    print("="*50)
    lstm = build_lstm((X_seq.shape[1], X_seq.shape[2]))
    hist = lstm.fit(
        X_seq_train, y_seq_train,
        validation_split=0.1,
        epochs=50, batch_size=32,
        callbacks=[early_stop], verbose=1
    )
    lstm.save("models/comparison_lstm.h5")
    y_pred_lstm       = lstm.predict(X_seq_test, verbose=0).flatten()
    results["LSTM"]   = compute_metrics(y_seq_test, y_pred_lstm, "LSTM")
    histories["LSTM"] = hist.history
    print(f"  ✅ LSTM MAE: {results['LSTM']['mae']:.4f}")

    # ── MODEL 2: GRU ──────────────────────────────────────
    print("\n" + "="*50)
    print("[2/5] Training GRU...")
    print("="*50)
    gru  = build_gru((X_seq.shape[1], X_seq.shape[2]))
    hist = gru.fit(
        X_seq_train, y_seq_train,
        validation_split=0.1,
        epochs=50, batch_size=32,
        callbacks=[early_stop], verbose=1
    )
    gru.save("models/comparison_gru.h5")
    y_pred_gru       = gru.predict(X_seq_test, verbose=0).flatten()
    results["GRU"]   = compute_metrics(y_seq_test, y_pred_gru, "GRU")
    histories["GRU"] = hist.history
    print(f"  ✅ GRU MAE: {results['GRU']['mae']:.4f}")

    # ── MODEL 3: XGBoost ──────────────────────────────────
    print("\n" + "="*50)
    print("[3/5] Training XGBoost...")
    print("="*50)
    xgb_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=6,
        learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, random_state=42,
        verbosity=0
    )
    xgb_model.fit(
        X_tab_train, y_tab_train,
        eval_set=[(X_tab_test, y_tab_test)],
        verbose=False
    )
    with open("models/comparison_xgb.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
    y_pred_xgb         = xgb_model.predict(X_tab_test)
    results["XGBoost"] = compute_metrics(y_tab_test, y_pred_xgb, "XGBoost")
    print(f"  ✅ XGBoost MAE: {results['XGBoost']['mae']:.4f}")

    # ── MODEL 4: Linear Regression ────────────────────────
    print("\n" + "="*50)
    print("[4/5] Training Linear Regression...")
    print("="*50)
    lr = LinearRegression()
    lr.fit(X_tab_train, y_tab_train)
    with open("models/comparison_lr.pkl", "wb") as f:
        pickle.dump(lr, f)
    y_pred_lr                   = lr.predict(X_tab_test)
    results["LinearRegression"] = compute_metrics(
        y_tab_test, y_pred_lr, "LinearRegression"
    )
    print(f"  ✅ LR MAE: {results['LinearRegression']['mae']:.4f}")

    # ── MODEL 5: Random Forest ────────────────────────────
    print("\n" + "="*50)
    print("[5/5] Training Random Forest...")
    print("="*50)
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=8,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_tab_train, y_tab_train)
    with open("models/comparison_rf.pkl", "wb") as f:
        pickle.dump(rf, f)
    y_pred_rf               = rf.predict(X_tab_test)
    results["RandomForest"] = compute_metrics(
        y_tab_test, y_pred_rf, "RandomForest"
    )
    print(f"  ✅ RF MAE: {results['RandomForest']['mae']:.4f}")

    predictions = {
        "LSTM"             : (y_seq_test, y_pred_lstm),
        "GRU"              : (y_seq_test, y_pred_gru),
        "XGBoost"          : (y_tab_test, y_pred_xgb),
        "LinearRegression" : (y_tab_test, y_pred_lr),
        "RandomForest"     : (y_tab_test, y_pred_rf),
    }

    return results, histories, predictions


def print_results_table(results):
    print("\n" + "="*65)
    print("  FINAL MODEL COMPARISON RESULTS")
    print("="*65)
    print(f"  {'Model':<22} {'MAE':>8} {'RMSE':>8} {'R²':>8}  Rank")
    print("-"*65)

    sorted_models = sorted(results.values(), key=lambda x: x["mae"])
    medals = ["🥇", "🥈", "🥉", "4️⃣ ", "5️⃣ "]

    for i, r in enumerate(sorted_models):
        print(f"  {r['model']:<22} {r['mae']:>8.4f} "
              f"{r['rmse']:>8.4f} {r['r2']:>8.4f}  {medals[i]}")

    print("="*65)
    winner = sorted_models[0]
    print(f"\n  🏆 WINNER: {winner['model']}")
    print(f"     MAE  = {winner['mae']:.4f}")
    print(f"     RMSE = {winner['rmse']:.4f}")
    print(f"     R²   = {winner['r2']:.4f}")
    print("="*65)

    pd.DataFrame(list(results.values())).to_csv(
        f"{RESULTS_DIR}/comparison_results.csv", index=False
    )
    print(f"\n  Results saved to {RESULTS_DIR}/comparison_results.csv")
    return sorted_models[0]["model"]


def generate_all_graphs(results, histories, predictions):
    print("\n[GRAPHS] Generating comparison graphs...")
    plt.style.use("dark_background")

    model_names = list(results.keys())
    mae_vals    = [results[m]["mae"]  for m in model_names]
    rmse_vals   = [results[m]["rmse"] for m in model_names]
    r2_vals     = [results[m]["r2"]   for m in model_names]
    colors      = [COLORS[m] for m in model_names]

    # Graph 1: MAE
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(model_names, mae_vals, color=colors,
                  edgecolor="white", linewidth=0.5)
    ax.set_title("Model Comparison — MAE (Lower is Better)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Mean Absolute Error (scaled)")
    ax.set_xlabel("Model")
    for bar, val in zip(bars, mae_vals):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.001,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    best_idx = mae_vals.index(min(mae_vals))
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(3)
    ax.text(best_idx, mae_vals[best_idx]/2, "BEST",
            ha="center", va="center",
            fontsize=11, fontweight="bold", color="gold")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/1_mae_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Graph 1 saved")

    # Graph 2: RMSE
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(model_names, rmse_vals, color=colors,
                  edgecolor="white", linewidth=0.5)
    ax.set_title("Model Comparison — RMSE (Lower is Better)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Root Mean Squared Error (scaled)")
    ax.set_xlabel("Model")
    for bar, val in zip(bars, rmse_vals):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.001,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    best_idx = rmse_vals.index(min(rmse_vals))
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/2_rmse_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Graph 2 saved")

    # Graph 3: R²
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(model_names, r2_vals, color=colors,
                  edgecolor="white", linewidth=0.5)
    ax.set_title("Model Comparison — R² Score (Higher is Better)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("R² Score (1.0 = Perfect)")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gold", linestyle="--", alpha=0.5)
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x()+bar.get_width()/2,
                max(bar.get_height()+0.01, 0.05),
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    best_idx = r2_vals.index(max(r2_vals))
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/3_r2_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Graph 3 saved")

    # Graph 4: Training Loss Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, name in enumerate(["LSTM", "GRU"]):
        ax   = axes[idx]
        hist = histories[name]
        ax.plot(hist["loss"],     color=COLORS[name],
                label="Train Loss", linewidth=2)
        ax.plot(hist["val_loss"], color="white",
                label="Val Loss",   linewidth=2, linestyle="--")
        ax.set_title(f"{name} Training Loss Curve",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.suptitle("Training Loss — LSTM vs GRU",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/4_training_loss_curves.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Graph 4 saved")

    # Graph 5: Prediction vs Actual
    best_model     = min(results, key=lambda m: results[m]["mae"])
    y_true, y_pred = predictions[best_model]
    show_n         = min(100, len(y_true))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(range(show_n), y_true[:show_n],
            color="white", label="Actual", linewidth=2)
    ax.plot(range(show_n), y_pred[:show_n],
            color=COLORS[best_model],
            label=f"{best_model} Prediction",
            linewidth=2, linestyle="--")
    ax.fill_between(range(show_n),
                    y_true[:show_n], y_pred[:show_n],
                    alpha=0.15, color=COLORS[best_model])
    ax.set_title(
        f"Prediction vs Actual — {best_model}\n"
        f"MAE={results[best_model]['mae']:.4f}  "
        f"R²={results[best_model]['r2']:.4f}",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Latency (scaled)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/5_prediction_vs_actual.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Graph 5 saved")

    # Graph 6: Full Summary
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(model_names, mae_vals, color=colors)
    ax1.set_title("MAE (Lower=Better)", fontweight="bold")
    ax1.set_xticklabels(model_names, rotation=15,
                        ha="right", fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(model_names, rmse_vals, color=colors)
    ax2.set_title("RMSE (Lower=Better)", fontweight="bold")
    ax2.set_xticklabels(model_names, rotation=15,
                        ha="right", fontsize=8)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(model_names, r2_vals, color=colors)
    ax3.set_title("R² Score (Higher=Better)", fontweight="bold")
    ax3.set_xticklabels(model_names, rotation=15,
                        ha="right", fontsize=8)
    ax3.set_ylim(0, 1.1)

    ax4 = fig.add_subplot(gs[1, :])
    ax4.plot(y_true[:show_n], color="white",
             label="Actual", linewidth=1.5)
    ax4.plot(y_pred[:show_n], color=COLORS[best_model],
             label=f"{best_model} (Best)",
             linewidth=1.5, linestyle="--")
    ax4.set_title(
        f"Best Model: {best_model} — Prediction vs Actual",
        fontweight="bold"
    )
    ax4.legend()
    ax4.grid(alpha=0.2)

    fig.suptitle(
        "Smart Network Latency Prediction — Model Comparison",
        fontsize=15, fontweight="bold"
    )
    plt.savefig(f"{RESULTS_DIR}/6_full_summary.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Graph 6 saved")


if __name__ == "__main__":
    print("="*65)
    print("  MODEL COMPARISON ENGINE")
    print("  Training 5 Models — Generating 6 Graphs")
    print("="*65)
    start_time = datetime.now()

    X_seq, y_seq, X_tab, y_tab, scaler, feature_cols = \
        load_and_prepare_data()

    results, histories, predictions = train_all_models(
        X_seq, y_seq, X_tab, y_tab
    )

    winner = print_results_table(results)

    generate_all_graphs(results, histories, predictions)

    elapsed = (datetime.now() - start_time).seconds
    print(f"\n[DONE] Completed in {elapsed} seconds!")
    print(f"[DONE] Winner: {winner}")
    print(f"[DONE] Graphs saved to: {RESULTS_DIR}/")
