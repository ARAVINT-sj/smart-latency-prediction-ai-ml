import subprocess
import time
import sqlite3
import csv
import os
import statistics
import psutil
import schedule
from datetime import datetime

TARGET_HOST    = "8.8.8.8"
PING_COUNT     = 10
INTERVAL_SEC   = 30
DB_FILE        = "data/network_metrics.db"
CSV_FILE       = "data/network_metrics.csv"

def setup_database():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS network_metrics (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            latency_ms      REAL,
            packet_loss     REAL,
            jitter_ms       REAL,
            throughput_mbps REAL,
            label           INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()
    print("[DB] Database ready!")

def ping_host(host, count=10):
    try:
        cmd = ["ping", "-n", str(count), host]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout
        rtts = []
        for line in output.split("\n"):
            if "time=" in line.lower():
                try:
                    part = line.lower().split("time=")[1]
                    rtt_str = part.split()[0].replace("ms","").strip()
                    rtts.append(float(rtt_str))
                except:
                    pass
        loss = 100.0
        for line in output.split("\n"):
            if "lost" in line.lower():
                try:
                    part = line.split("(")[1]
                    loss = float(part.split("%")[0])
                    break
                except:
                    pass
        return rtts, loss
    except Exception as e:
        print(f"[ERROR] Ping failed: {e}")
        return [], 100.0

def get_throughput_mbps():
    try:
        net1 = psutil.net_io_counters()
        time.sleep(1)
        net2 = psutil.net_io_counters()
        bytes_per_sec = (net2.bytes_recv - net1.bytes_recv) + \
                        (net2.bytes_sent - net1.bytes_sent)
        return round((bytes_per_sec * 8) / 1_000_000, 4)
    except:
        return 0.0

def assign_label(latency, packet_loss):
    if latency is None:
        return 2
    if latency < 50 and packet_loss < 1:
        return 0
    elif latency < 150 and packet_loss < 5:
        return 1
    else:
        return 2

def collect_metrics():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] Collecting...")
    rtts, packet_loss = ping_host(TARGET_HOST, PING_COUNT)
    if rtts:
        latency_ms = round(statistics.mean(rtts), 3)
        jitter_ms  = round(statistics.stdev(rtts) if len(rtts) > 1 else 0.0, 3)
    else:
        latency_ms = None
        jitter_ms  = None
        packet_loss = 100.0
    throughput = get_throughput_mbps()
    label = assign_label(latency_ms, packet_loss)
    label_names = {0:"Normal", 1:"Degraded", 2:"Critical"}
    print(f"  Latency   : {latency_ms} ms")
    print(f"  Jitter    : {jitter_ms} ms")
    print(f"  Pkt Loss  : {packet_loss} %")
    print(f"  Throughput: {throughput} Mbps")
    print(f"  Status    : {label_names[label]}")
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        INSERT INTO network_metrics
        (timestamp,latency_ms,packet_loss,jitter_ms,throughput_mbps,label)
        VALUES (?,?,?,?,?,?)
    """, (timestamp, latency_ms, packet_loss, jitter_ms, throughput, label))
    conn.commit()
    conn.close()
    csv_exists = os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(["timestamp","latency_ms","packet_loss",
                             "jitter_ms","throughput_mbps","label"])
        writer.writerow([timestamp,latency_ms,packet_loss,
                         jitter_ms,throughput,label])
    print("  [SAVED]✅")

if __name__ == "__main__":
    print("="*50)
    print(" Network Collector Starting...")
    print("="*50)
    setup_database()
    collect_metrics()
    schedule.every(INTERVAL_SEC).seconds.do(collect_metrics)
    print(f"\nCollecting every {INTERVAL_SEC}s. Press Ctrl+C to stop.\n")
    while True:
        schedule.run_pending()
        time.sleep(1)


