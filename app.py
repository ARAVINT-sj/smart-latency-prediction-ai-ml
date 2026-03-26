from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import sqlite3
import os
from datetime import datetime, timedelta
import threading
import time

app = Flask(__name__)
CORS(app)

DB_FILE   = "data/network_metrics.db"
ALERTS_DB = "data/alerts.db"

def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def setup_alerts_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(ALERTS_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            severity  TEXT,
            message   TEXT,
            latency   REAL,
            resolved  INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()

def save_alert(severity, message, latency):
    conn = sqlite3.connect(ALERTS_DB)
    conn.execute("""
        INSERT INTO alerts (timestamp, severity, message, latency)
        VALUES (?, ?, ?, ?)
    """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          severity, message, latency))
    conn.commit()
    conn.close()

@app.route("/")
def index():
    return render_template("dashboard.html")
@app.route("/graphs/<filename>")
def serve_graph(filename):
    from flask import send_from_directory
    return send_from_directory(
        "data/comparison_results", filename
    )

@app.route("/api/metrics/live")
def get_live_metrics():
    try:
        conn = get_db()
        rows = conn.execute("""
            SELECT timestamp, latency_ms, packet_loss,
                   jitter_ms, throughput_mbps, label
            FROM network_metrics
            ORDER BY timestamp DESC
            LIMIT 50
        """).fetchall()
        conn.close()
        data = [dict(row) for row in reversed(rows)]
        return jsonify({"status":"ok", "data":data, "count":len(data)})
    except Exception as e:
        return jsonify({"status":"error", "message":str(e)}), 500

@app.route("/api/metrics/history")
def get_history():
    hours = request.args.get("hours", 24, type=int)
    since = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    try:
        conn = get_db()
        rows = conn.execute("""
            SELECT timestamp, latency_ms, packet_loss,
                   jitter_ms, throughput_mbps, label
            FROM network_metrics
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
        """, (since,)).fetchall()
        conn.close()
        data = [dict(row) for row in rows]
        return jsonify({"status":"ok", "data":data})
    except Exception as e:
        return jsonify({"status":"error", "message":str(e)}), 500

@app.route("/api/metrics/summary")
def get_summary():
    try:
        conn = get_db()
        row = conn.execute("""
            SELECT
                ROUND(AVG(latency_ms),2)      AS avg_latency,
                ROUND(MIN(latency_ms),2)      AS min_latency,
                ROUND(MAX(latency_ms),2)      AS max_latency,
                ROUND(AVG(packet_loss),2)     AS avg_loss,
                ROUND(AVG(jitter_ms),2)       AS avg_jitter,
                ROUND(AVG(throughput_mbps),4) AS avg_throughput,
                COUNT(*)                       AS total_samples
            FROM network_metrics
            WHERE timestamp >= datetime('now','-24 hours')
        """).fetchone()
        conn.close()
        return jsonify({"status":"ok", "data":dict(row)})
    except Exception as e:
        return jsonify({"status":"error", "message":str(e)}), 500

@app.route("/api/predict/latency")
def predict_latency():
    try:
        from lstm_model import predict_future_latency
        predictions = predict_future_latency(steps_ahead=[1,2,3])
        if predictions:
            return jsonify({"status":"ok", "predictions":predictions})
        else:
            return jsonify({
                "status":"no_model",
                "message":"Train LSTM first. Run lstm_model.py"
            })
    except Exception as e:
        return jsonify({"status":"error", "message":str(e)}), 500

@app.route("/api/predict/health")
def predict_health():
    try:
        from rf_classifier import predict_health_state
        result = predict_health_state()
        if result.get("label") == 2:
            save_alert(
                "CRITICAL",
                "Network Critical: " + result.get("recommendation",""),
                result.get("avg_latency_ms", 0)
            )
        elif result.get("label") == 1:
            save_alert(
                "WARNING",
                "Network Degraded: " + result.get("recommendation",""),
                result.get("avg_latency_ms", 0)
            )
        return jsonify({"status":"ok", "health":result})
    except Exception as e:
        return jsonify({"status":"error", "message":str(e)}), 500

@app.route("/api/alerts")
def get_alerts():
    limit = request.args.get("limit", 20, type=int)
    try:
        conn = sqlite3.connect(ALERTS_DB)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT * FROM alerts
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,)).fetchall()
        conn.close()
        data = [dict(row) for row in rows]
        return jsonify({"status":"ok", "alerts":data})
    except Exception as e:
        return jsonify({"status":"error", "message":str(e)}), 500

@app.route("/api/status")
def api_status():
    try:
        lstm_ready = os.path.exists("models/lstm_model.h5")
        rf_ready   = os.path.exists("models/rf_classifier.pkl")
        conn = get_db()
        count = conn.execute(
            "SELECT COUNT(*) as cnt FROM network_metrics"
        ).fetchone()["cnt"]
        conn.close()
        return jsonify({
            "status": "running",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "lstm_model_ready": lstm_ready,
            "rf_model_ready":   rf_ready,
            "total_samples_collected": count
        })
    except Exception as e:
        return jsonify({"status":"error", "message":str(e)}), 500

def background_alert_checker():
    while True:
        try:
            conn = get_db()
            row  = conn.execute("""
                SELECT latency_ms, packet_loss
                FROM network_metrics
                ORDER BY timestamp DESC LIMIT 1
            """).fetchone()
            conn.close()
            if row:
                latency = row["latency_ms"] or 0
                loss    = row["packet_loss"] or 0
                if latency > 200:
                    save_alert("CRITICAL",
                        f"Latency spike! {latency:.1f}ms detected!",
                        latency)
                elif latency > 100:
                    save_alert("WARNING",
                        f"High latency: {latency:.1f}ms detected!",
                        latency)
                if loss > 10:
                    save_alert("CRITICAL",
                        f"High packet loss: {loss:.1f}%!",
                        latency)
        except:
            pass
        time.sleep(60)

if __name__ == "__main__":
    print("="*50)
    print("  Network Quality Prediction System")
    print("="*50)
    setup_alerts_db()
    alert_thread = threading.Thread(
        target=background_alert_checker, daemon=True
    )
    alert_thread.start()
    print("\n[INFO] Open browser at: http://localhost:5000")
    print("[INFO] Press Ctrl+C to stop\n")
    app.run(host="0.0.0.0", port=5000, debug=False)