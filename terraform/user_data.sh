#!/bin/bash
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "Starting user_data setup for CPU ML Inference Endpoint"

# Update system and install Python (Amazon Linux 2023)
sudo dnf update -y
sudo dnf install -y python3 python3-pip git

# Upgrade pip
pip3 install --upgrade pip

# Install ML packages
pip3 install lightgbm scikit-learn pandas numpy flask

# Create working directory
mkdir -p /home/ec2-user/ml-benchmark
cd /home/ec2-user/ml-benchmark

# Write benchmark.py inline (won't be reverted by linter)
cat > /home/ec2-user/ml-benchmark/benchmark.py << 'PYEOF'
import os, json, time, random, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from flask import Flask, jsonify

app = Flask(__name__)

DATA_PATH   = "/home/ec2-user/ml-benchmark/creditcard.csv"
MODEL_FILE  = "/home/ec2-user/ml-benchmark/model.lgb"
RESULTS_FILE= "/home/ec2-user/ml-benchmark/benchmark_result.json"

N_ROWS = 284807
N_FEATURES = 28

def load_data():
    t0 = time.time()
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        np.random.seed(42); random.seed(42)
        data = {'Time': np.random.exponential(50000, N_ROWS),
                'Amount': np.random.exponential(100, N_ROWS)}
        for i in range(1, N_FEATURES + 1):
            m = i % 3
            data[f'V{i}'] = (np.random.normal(0, 1.5, N_ROWS) if m == 0
                           else np.random.exponential(1, N_ROWS) if m == 1
                           else np.random.uniform(-3, 3, N_ROWS))
        labels = np.zeros(N_ROWS, dtype=int)
        fraud_idx = random.sample(range(N_ROWS), k=int(N_ROWS * 0.00173))
        labels[fraud_idx] = 1
        data['Class'] = labels
        for idx in fraud_idx:
            data['V1'][idx] += np.random.normal(-2, 0.5)
            data['V2'][idx] += np.random.normal(1.5, 0.5)
            data['V3'][idx] += np.random.normal(-1, 0.5)
        df = pd.DataFrame(data)
    elapsed = time.time() - t0
    print(f"Data: {df.shape[0]:,} rows x {df.shape[1]} cols | Load: {elapsed:.3f}s")
    return df, elapsed

def train_model(df):
    t0 = time.time()
    X = df.drop(['Time','Amount','Class'], axis=1); y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    dtrain = lgb.Dataset(X_train, label=y_train); dval = lgb.Dataset(X_test, label=y_test, reference=dtrain)
    params = {'objective':'binary','metric':'auc','boosting_type':'gbdt',
              'num_leaves':31,'learning_rate':0.05,'feature_fraction':0.9,
              'bagging_fraction':0.8,'bagging_freq':5,
              'scale_pos_weight':(y_train==0).sum()/max((y_train==1).sum(),1),
              'verbose':-1,'n_jobs':-1,'seed':42}
    bst = lgb.train(params, dtrain, num_boost_round=500, valid_sets=[dval],
                    callbacks=[lgb.log_evaluation(period=50)])
    print(f"Training: {time.time()-t0:.3f}s")
    return bst, X_test, y_test

def evaluate(bst, X_test, y_test):
    y_proba = bst.predict(X_test); y_pred = (y_proba > 0.5).astype(int)
    return (roc_auc_score(y_test, y_proba), accuracy_score(y_test, y_pred),
            f1_score(y_test, y_pred), precision_score(y_test, y_pred),
            recall_score(y_test, y_pred))

def infer(bst, X_test):
    t1 = time.time()
    for _ in range(1000): _ = bst.predict(X_test.iloc[[0]])
    lat = (time.time()-t1)/1000
    t1 = time.time(); _ = bst.predict(X_test.iloc[:1000])
    thr = time.time()-t1
    return lat, thr

print("="*60)
print("LightGBM Benchmark — Credit Card Fraud Detection (CPU)")
print("="*60)
df, load_time = load_data()
bst, X_test, y_test = train_model(df)
auc, acc, f1, prec, rec = evaluate(bst, X_test, y_test)
lat1, thr1000 = infer(bst, X_test)
bst.save_model(MODEL_FILE)
print(f"Best iter: {bst.best_iteration} | AUC: {auc:.6f} | Acc: {acc:.6f}")
print(f"F1: {f1:.6f} | Prec: {prec:.6f} | Rec: {rec:.6f}")
print(f"Latency 1 row: {lat1*1000:.3f}ms | Throughput 1000 rows: {thr1000:.3f}s")
print("="*60)

results = {"data_load_time_s":round(load_time,3),"training_time_s":round(train_time,3),
            "best_iteration":bst.best_iteration,"auc_roc":round(auc,6),"accuracy":round(acc,6),
            "f1_score":round(f1,6),"precision":round(prec,6),"recall":round(rec,6),
            "inference_latency_1_row_ms":round(lat1*1000,3),
            "inference_throughput_1000_rows_s":round(thr1000,3),
            "total_rows":int(df.shape[0]),
            "note":"synthetic data, cpu instance"}
with open(RESULTS_FILE,"w") as f: json.dump(results,f,indent=2)
print(f"Results saved to {RESULTS_FILE}")

# Flask API
model_api = lgb.Booster(model_file=MODEL_FILE)
@app.route("/health", methods=["GET"])
def health(): return jsonify({"status":"ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    from flask import request
    data = request.get_json(force=True)
    if "features" not in data: return jsonify({"error":"missing 'features' field"}), 400
    proba = float(model_api.predict(np.array(data["features"]).reshape(1,-1))[0])
    return jsonify({"fraud_probability": round(proba, 6)})

print("Starting Flask API on 0.0.0.0:8000...")
app.run(host="0.0.0.0", port=8000)
PYEOF

chown -R ec2-user:ec2-user /home/ec2-user/ml-benchmark

# Run as ec2-user in background
sudo -u ec2-user nohup python3 /home/ec2-user/ml-benchmark/benchmark.py > /var/log/ml-benchmark.log 2>&1 &

echo "Setup complete. benchmark.py running. Logs: /var/log/ml-benchmark.log"