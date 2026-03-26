import numpy as np
import pandas as pd
import pickle
import os
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

RF_MODEL_FILE = "models/rf_classifier.pkl"

def load_classification_data():
    print("[RF] Loading data...")
    conn = sqlite3.connect("data/network_metrics.db")
    df = pd.read_sql("SELECT * FROM network_metrics ORDER BY timestamp", conn)
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.fillna(0)
    window = 10
    df["avg_latency"]    = df["latency_ms"].rolling(window, min_periods=1).mean()
    df["avg_pkt_loss"]   = df["packet_loss"].rolling(window, min_periods=1).mean()
    df["avg_jitter"]     = df["jitter_ms"].rolling(window, min_periods=1).mean()
    df["avg_throughput"] = df["throughput_mbps"].rolling(window, min_periods=1).mean()
    df["latency_std"]    = df["latency_ms"].rolling(window, min_periods=1).std().fillna(0)
    df["hour"]           = df["timestamp"].dt.hour
    df["is_peak"]        = df["hour"].apply(
        lambda h: 1 if (8 <= h <= 10) or (18 <= h <= 21) else 0
    )
    df = df.dropna(subset=["label"])
    feature_cols = [
        "avg_latency", "avg_pkt_loss", "avg_jitter",
        "avg_throughput", "latency_std", "hour", "is_peak"
    ]
    X = df[feature_cols].values
    y = df["label"].values.astype(int)
    print(f"  Total samples: {len(df)}")
    label_names = {0:"Normal", 1:"Degraded", 2:"Critical"}
    for label, name in label_names.items():
        count = (y == label).sum()
        print(f"  {name}: {count} samples ({count/len(y)*100:.1f}%)")
    return X, y, feature_cols

def train_random_forest(X, y):
    print("\n[RF] Training Random Forest...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print("  Training complete!")
    return rf, X_test, y_test

def evaluate_classifier(rf, X_test, y_test, feature_cols):
    y_pred = rf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print("\n" + "="*50)
    print("  Random Forest Results")
    print("="*50)
    print(f"\n  Overall Accuracy: {acc*100:.2f}%\n")
    print("  Classification Report:")
    unique_labels     = sorted(list(set(y_test) | set(y_pred)))
    label_map         = {0:"Normal", 1:"Degraded", 2:"Critical"}
    target_names_used = [label_map[l] for l in unique_labels]
    print(classification_report(
        y_test, y_pred,
        labels=unique_labels,
        target_names=target_names_used
    ))
    print("  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    for i, row in enumerate(cm):
        print(f"  {label_map[unique_labels[i]]:10s}: {row}")
    print("\n  Feature Importance:")
    importances = rf.feature_importances_
    sorted_idx  = np.argsort(importances)[::-1]
    for i in sorted_idx:
        bar = "█" * int(importances[i] * 40)
        print(f"  {feature_cols[i]:20s}: {bar} ({importances[i]*100:.1f}%)")
    print("="*50)
    return acc

def predict_health_state():
    try:
        with open(RF_MODEL_FILE, "rb") as f:
            rf = pickle.load(f)
        conn = sqlite3.connect("data/network_metrics.db")
        df   = pd.read_sql(
            "SELECT * FROM network_metrics ORDER BY timestamp DESC LIMIT 10",
            conn
        )
        conn.close()
        if len(df) < 5:
            return {
                "state": "Unknown",
                "label": -1,
                "probabilities": [0,0,0],
                "recommendation": "Not enough data yet"
            }
        df             = df.sort_values("timestamp")
        avg_latency    = df["latency_ms"].mean()
        avg_pkt_loss   = df["packet_loss"].mean()
        avg_jitter     = df["jitter_ms"].mean()
        avg_throughput = df["throughput_mbps"].mean()
        latency_std    = df["latency_ms"].std()
        hour           = pd.Timestamp.now().hour
        is_peak        = 1 if (8<=hour<=10) or (18<=hour<=21) else 0
        features = np.array([[
            avg_latency, avg_pkt_loss, avg_jitter,
            avg_throughput, latency_std, hour, is_peak
        ]])
        label = int(rf.predict(features)[0])
        probs = rf.predict_proba(features)[0].tolist()
        state_names = {0:"Normal", 1:"Degraded", 2:"Critical"}
        recommendations = {
            0: "Network is healthy. No action needed.",
            1: "Network degrading. Monitor closely.",
            2: "CRITICAL! High latency detected. Act immediately!"
        }
        return {
            "state": state_names[label],
            "label": label,
            "probabilities": [round(p*100,1) for p in probs],
            "avg_latency_ms": round(avg_latency, 2),
            "avg_packet_loss": round(avg_pkt_loss, 2),
            "recommendation": recommendations[label]
        }
    except FileNotFoundError:
        return {
            "state": "Model not trained yet",
            "label": -1,
            "probabilities": [0,0,0],
            "recommendation": "Run rf_classifier.py first"
        }
    except Exception as e:
        return {
            "state": "Error",
            "label": -1,
            "probabilities": [0,0,0],
            "recommendation": str(e)
        }

if __name__ == "__main__":
    print("="*55)
    print("  Random Forest Network Health Classifier")
    print("="*55)
    X, y, feature_cols = load_classification_data()
    if len(X) < 30:
        print("[ERROR] Need at least 30 samples!")
        exit()
    print("\n[RF] Running 5-fold cross validation...")
    rf_cv = RandomForestClassifier(
        n_estimators=100, max_depth=8,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    scores = cross_val_score(rf_cv, X, y, cv=5, scoring="accuracy")
    print(f"  CV Accuracy: {scores.mean()*100:.2f}% +/- {scores.std()*100:.2f}%")
    rf, X_test, y_test = train_random_forest(X, y)
    evaluate_classifier(rf, X_test, y_test, feature_cols)
    os.makedirs("models", exist_ok=True)
    with open(RF_MODEL_FILE, "wb") as f:
        pickle.dump(rf, f)
    print(f"\n[SAVED] RF model saved!")
    print("\n[TEST] Current network health:")
    result = predict_health_state()
    for k, v in result.items():
        print(f"  {k}: {v}")
