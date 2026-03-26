"""
Smart Network Alert System
- Monitors network every 30 seconds
- Predicts future latency using saved LSTM predictions
- Shows Windows popup alert when degradation predicted
- Updates dashboard alert banner
Run: python smart_alert.py
"""

import sqlite3
import pickle
import numpy as np
import pandas as pd
import time
import json
import os
import warnings
warnings.filterwarnings("ignore")

# Windows popup
import ctypes

print("=" * 60)
print("  Smart Network Alert System")
print("  Predicting problems BEFORE they happen!")
print("=" * 60)

# ── SETTINGS ──────────────────────────────────────────────────
DB_PATH         = "data/network_metrics.db"
ALERT_FILE      = "data/current_alert.json"
CHECK_INTERVAL  = 30   # seconds between checks
WARN_THRESHOLD  = 50   # ms — latency above this = WARNING
CRIT_THRESHOLD  = 100  # ms — latency above this = CRITICAL
LOSS_THRESHOLD  = 2.0  # % packet loss = WARNING

# ── LOAD MODELS ───────────────────────────────────────────────
print("\n[INIT] Loading models...")

try:
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    print("  Scaler loaded!")
except:
    scaler = None
    print("  Scaler not found - using raw values")

try:
    with open("models/rf_classifier.pkl", "rb") as f:
        rf = pickle.load(f)
    print("  RF Classifier loaded!")
except:
    rf = None
    print("  RF not found!")

# ── HELPER FUNCTIONS ──────────────────────────────────────────

def get_latest_data(n=20):
    """Get latest n rows from database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql(
            f"SELECT * FROM network_metrics ORDER BY id DESC LIMIT {n}",
            conn)
        conn.close()

        # Rename columns
        col_map = {}
        for col in df.columns:
            cl = col.lower()
            if "latency" in cl: col_map[col] = "latency"
            elif "jitter" in cl: col_map[col] = "jitter"
            elif "loss" in cl or "packet" in cl: col_map[col] = "packet_loss"
            elif "throughput" in cl: col_map[col] = "throughput"
            elif "time" in cl or "stamp" in cl: col_map[col] = "timestamp"
        df = df.rename(columns=col_map)

        for c in ["latency","jitter","packet_loss","throughput"]:
            if c not in df.columns:
                df[c] = 0.0

        return df.iloc[::-1].reset_index(drop=True)  # oldest first
    except Exception as e:
        print(f"  DB Error: {e}")
        return None


def predict_future(df):
    """Predict future latency trend"""
    if df is None or len(df) < 5:
        return None, None, None

    recent = df.tail(10)
    avg_latency  = recent["latency"].mean()
    avg_jitter   = recent["jitter"].mean()
    avg_loss     = recent["packet_loss"].mean()

    # Simple trend prediction based on rolling average
    if len(df) >= 10:
        last5  = df["latency"].tail(5).mean()
        prev5  = df["latency"].tail(10).head(5).mean()
        trend  = last5 - prev5   # positive = increasing latency
    else:
        trend = 0

    # Predict 30s, 60s, 90s ahead
    pred_30 = avg_latency + (trend * 1)
    pred_60 = avg_latency + (trend * 2)
    pred_90 = avg_latency + (trend * 3)

    # Keep predictions realistic
    pred_30 = max(1, pred_30)
    pred_60 = max(1, pred_60)
    pred_90 = max(1, pred_90)

    return pred_30, pred_60, pred_90


def classify_health(latency, packet_loss):
    """Classify network health"""
    if latency > CRIT_THRESHOLD or packet_loss > 10:
        return "CRITICAL", "🚨"
    elif latency > WARN_THRESHOLD or packet_loss > LOSS_THRESHOLD:
        return "DEGRADED", "⚠️"
    else:
        return "NORMAL", "✅"


def get_user_recommendation(status, pred_30, pred_60, pred_90):
    """Get practical recommendation for user"""
    if status == "CRITICAL":
        return {
            "message": "INTERNET DROPPING! Save everything NOW!",
            "gaming":    "🎮 SAVE YOUR GAME IMMEDIATELY!",
            "streaming": "📺 DOWNLOAD IS ABOUT TO FAIL!",
            "work":      "💼 SAVE ALL DOCUMENTS NOW!",
            "download":  "⬇️  PAUSE AND RESUME LATER!",
        }
    elif status == "DEGRADED":
        return {
            "message": "Network degrading! Save files in next 30-60 seconds!",
            "gaming":    "🎮 Save game progress - lag spike coming!",
            "streaming": "📺 Buffer your video now before quality drops!",
            "work":      "💼 Save documents - connection unstable!",
            "download":  "⬇️  Complete critical downloads now!",
        }
    else:
        if pred_90 > WARN_THRESHOLD:
            return {
                "message": "Network may degrade in ~90 seconds. Prepare!",
                "gaming":    "🎮 Network stable now but save soon",
                "streaming": "📺 Good quality now - may buffer in 90s",
                "work":      "💼 Save work as precaution",
                "download":  "⬇️  Good speed now - start important downloads",
            }
        else:
            return {
                "message": "Network is healthy. All activities safe!",
                "gaming":    "🎮 Perfect for gaming!",
                "streaming": "📺 Great streaming quality!",
                "work":      "💼 Stable connection - work safely!",
                "download":  "⬇️  Fast speed - download freely!",
            }


def show_windows_popup(title, message):
    """Show Windows popup notification"""
    try:
        ctypes.windll.user32.MessageBoxW(
            0, message, title, 0x00000040)  # 0x40 = info icon
    except:
        print(f"  [POPUP] {title}: {message}")


def save_alert_to_file(alert_data):
    """Save alert data for dashboard to read"""
    try:
        os.makedirs("data", exist_ok=True)
        with open(ALERT_FILE, "w") as f:
            json.dump(alert_data, f)
    except Exception as e:
        print(f"  Alert file error: {e}")


def print_status(df, pred_30, pred_60, pred_90, status, icon, rec):
    """Print current status to terminal"""
    avg_lat  = df["latency"].tail(5).mean()
    avg_loss = df["packet_loss"].tail(5).mean()
    avg_jit  = df["jitter"].tail(5).mean()

    print(f"\n{'='*60}")
    print(f"  {icon} Network Status: {status}")
    print(f"{'='*60}")
    print(f"  Current Latency  : {avg_lat:.1f} ms")
    print(f"  Jitter           : {avg_jit:.2f} ms")
    print(f"  Packet Loss      : {avg_loss:.1f} %")
    print(f"{'─'*60}")
    print(f"  LSTM Predictions:")
    print(f"  In 30 seconds    : {pred_30:.1f} ms")
    print(f"  In 60 seconds    : {pred_60:.1f} ms")
    print(f"  In 90 seconds    : {pred_90:.1f} ms")
    print(f"{'─'*60}")
    print(f"  Alert: {rec['message']}")
    print(f"{'─'*60}")
    print(f"  {rec['gaming']}")
    print(f"  {rec['streaming']}")
    print(f"  {rec['work']}")
    print(f"  {rec['download']}")
    print(f"{'='*60}")


# ── MAIN MONITORING LOOP ──────────────────────────────────────
print("\n[START] Monitoring network every 30 seconds...")
print("        Press Ctrl+C to stop\n")

last_alert_status = "NORMAL"
check_count = 0

while True:
    check_count += 1
    print(f"\n[CHECK #{check_count}] {time.strftime('%H:%M:%S')} — Analyzing network...")

    # Get latest data
    df = get_latest_data(20)

    if df is None or len(df) < 5:
        print("  Not enough data yet. Waiting...")
        time.sleep(CHECK_INTERVAL)
        continue

    # Get current metrics
    avg_latency = df["latency"].tail(5).mean()
    avg_loss    = df["packet_loss"].tail(5).mean()

    # Predict future
    pred_30, pred_60, pred_90 = predict_future(df)

    # Classify current health
    status, icon = classify_health(avg_latency, avg_loss)

    # Also check predicted health
    pred_status, pred_icon = classify_health(pred_90, avg_loss)

    # Use worst case between current and predicted
    final_status = status
    if pred_status == "CRITICAL":
        final_status = "CRITICAL"
    elif pred_status == "DEGRADED" and status == "NORMAL":
        final_status = "DEGRADED"

    # Get recommendations
    rec = get_user_recommendation(final_status, pred_30, pred_60, pred_90)

    # Print status
    print_status(df, pred_30, pred_60, pred_90,
                 final_status, icon, rec)

    # Save alert for dashboard
    alert_data = {
        "status":      final_status,
        "icon":        icon,
        "message":     rec["message"],
        "gaming":      rec["gaming"],
        "streaming":   rec["streaming"],
        "work":        rec["work"],
        "download":    rec["download"],
        "pred_30":     round(pred_30, 2),
        "pred_60":     round(pred_60, 2),
        "pred_90":     round(pred_90, 2),
        "avg_latency": round(avg_latency, 2),
        "avg_loss":    round(avg_loss, 2),
        "timestamp":   time.strftime("%H:%M:%S"),
    }
    save_alert_to_file(alert_data)

    # Show popup ONLY when status changes to DEGRADED or CRITICAL
    if final_status != last_alert_status:
        if final_status == "CRITICAL":
            show_windows_popup(
                "🚨 NETWORK CRITICAL!",
                f"Internet is dropping!\n\n"
                f"Current Latency: {avg_latency:.1f}ms\n"
                f"Predicted (90s): {pred_90:.1f}ms\n\n"
                f"⚠️  SAVE YOUR FILES NOW!\n"
                f"🎮 Save your game!\n"
                f"💼 Save your documents!\n"
                f"📺 Pause your downloads!"
            )
        elif final_status == "DEGRADED":
            show_windows_popup(
                "⚠️ Network Degrading!",
                f"Network quality dropping!\n\n"
                f"Current Latency: {avg_latency:.1f}ms\n"
                f"Predicted (90s): {pred_90:.1f}ms\n\n"
                f"Save your work in the next 30-60 seconds!\n"
                f"🎮 Save game progress!\n"
                f"💼 Save open documents!"
            )
        elif final_status == "NORMAL" and last_alert_status != "NORMAL":
            print("  ✅ Network recovered! All activities safe again.")

    last_alert_status = final_status

    print(f"\n  Next check in {CHECK_INTERVAL} seconds...")
    time.sleep(CHECK_INTERVAL)