import sqlite3, pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

conn = sqlite3.connect("data/network_metrics.db")
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cur.fetchall()
print("Tables:", tables)
table_name = tables[0][0]
df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
conn.close()
print("Rows:", len(df))

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
    if c not in df.columns: df[c] = 0.0

df = df.dropna(subset=["latency"])
df = df[(df["latency"]>0)&(df["latency"]<500)].reset_index(drop=True)
print("Clean rows:", len(df))

if "timestamp" in df.columns:
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_peak_hour"] = df["hour"].apply(lambda h: 1 if (8<=h<=10 or 18<=h<=22) else 0)
    except:
        df["hour"]=12; df["day_of_week"]=0; df["is_peak_hour"]=0
else:
    df["hour"]=12; df["day_of_week"]=0; df["is_peak_hour"]=0

df["rolling_avg_latency"] = df["latency"].rolling(5,min_periods=1).mean()
df["rolling_std_latency"] = df["latency"].rolling(5,min_periods=1).std().fillna(0)
df["rolling_avg_jitter"]  = df["jitter"].rolling(5,min_periods=1).mean()
df["rolling_avg_loss"]    = df["packet_loss"].rolling(5,min_periods=1).mean()

FEATURES = ["latency","jitter","packet_loss","throughput","hour","day_of_week","is_peak_hour","rolling_avg_latency","rolling_std_latency","rolling_avg_jitter","rolling_avg_loss"]
LABELS   = ["Latency","Jitter","Packet Loss","Throughput","Hour","Day of Week","Peak Hour","Roll Avg Latency","Roll Std Latency","Roll Avg Jitter","Roll Avg Loss"]

X = df[FEATURES].values
print("Shape:", X.shape)

with open("models/rf_classifier.pkl","rb") as f: rf = pickle.load(f)
print("RF loaded!")

print("Running SHAP - wait 1-2 minutes...")
X_sample = X[:300]
X_df = pd.DataFrame(X_sample, columns=LABELS)
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_sample)

if isinstance(shap_values, list):
    sv = np.array(shap_values[0])
else:
    sv = np.array(shap_values)
sv = sv[:, :len(LABELS)]
print("SHAP done!")
print("SV shape:", sv.shape)

plt.figure(figsize=(10,7))
shap.summary_plot(sv, X_df, plot_type="bar", show=False)
plt.title("SHAP Feature Importance", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("shap_1_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_1_feature_importance.png")

plt.figure(figsize=(10,7))
shap.summary_plot(sv, X_df, show=False)
plt.title("SHAP Summary Plot", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("shap_2_summary_dot.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_2_summary_dot.png")

mean_shap = np.abs(sv).mean(axis=0).flatten()[:len(LABELS)]
sorted_idx = [int(i) for i in np.argsort(mean_shap)]
sorted_vals = [float(mean_shap[i]) for i in sorted_idx]
sorted_labs = [LABELS[i] for i in sorted_idx]

fig, ax = plt.subplots(figsize=(10,7))
ax.barh(sorted_labs, sorted_vals, color="#58a6ff")
ax.set_xlabel("Mean |SHAP Value|")
ax.set_title("SHAP Feature Importance", fontweight="bold")
plt.tight_layout()
plt.savefig("shap_3_custom_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_3_custom_bar.png")

print("="*60)
print("SHAP COMPLETE!")
for rank, idx in enumerate(sorted_idx[::-1], 1):
    print(f"  {rank}. {LABELS[idx]:<25} {float(mean_shap[idx]):.4f}")
print("="*60)