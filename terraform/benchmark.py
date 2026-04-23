"""
LightGBM Benchmark — Credit Card Fraud Detection
Standalone script: generates synthetic data (mimics real creditcard.csv)
Runs: train → evaluate → inference → save results → expose Flask API
"""
import os
import json
import time
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from flask import Flask, jsonify

app = Flask(__name__)

DATA_PATH = "/home/ec2-user/ml-benchmark/creditcard.csv"
N_ROWS = 284807
N_FEATURES = 28
MODEL_FILE = "/home/ec2-user/ml-benchmark/model.lgb"
RESULTS_FILE = "/home/ec2-user/ml-benchmark/benchmark_result.json"

# ---------------------------------------------------------------------------
# 1. Load or generate data
# ---------------------------------------------------------------------------
def load_data():
    t0 = time.time()
    if os.path.exists(DATA_PATH):
        print(f"Loading dataset from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)
    else:
        print(f"Dataset not found. Generating synthetic data ({N_ROWS:,} rows, {N_FEATURES} features)...")
        np.random.seed(42)
        random.seed(42)

        # Simulate credit-card-like features (V1–V28, Time, Amount, Class)
        data = {}
        data['Time'] = np.random.exponential(50000, N_ROWS)
        data['Amount'] = np.random.exponential(100, N_ROWS)

        for i in range(1, N_FEATURES + 1):
            # Mix of distributions to mimic real V features
            if i % 3 == 0:
                data[f'V{i}'] = np.random.normal(0, 1.5, N_ROWS)
            elif i % 3 == 1:
                data[f'V{i}'] = np.random.exponential(1, N_ROWS)
            else:
                data[f'V{i}'] = np.random.uniform(-3, 3, N_ROWS)

        # Highly imbalanced: ~0.173% fraud
        labels = np.zeros(N_ROWS, dtype=int)
        fraud_idx = random.sample(range(N_ROWS), k=int(N_ROWS * 0.00173))
        labels[fraud_idx] = 1
        data['Class'] = labels

        # Add small fraud signal to some features
        for idx in fraud_idx:
            data['V1'][idx] += np.random.normal(-2, 0.5)
            data['V2'][idx] += np.random.normal(1.5, 0.5)
            data['V3'][idx] += np.random.normal(-1, 0.5)

        df = pd.DataFrame(data)

    elapsed = time.time() - t0
    print(f"Data loaded: {df.shape[0]:,} rows x {df.shape[1]} columns | Load time: {elapsed:.3f}s")
    return df, elapsed

# ---------------------------------------------------------------------------
# 2. Train LightGBM
# ---------------------------------------------------------------------------
def train_model(df):
    t0 = time.time()
    X = df.drop(['Time', 'Amount', 'Class'], axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval   = lgb.Dataset(X_test, label=y_test, reference=dtrain)

    params = {
        'objective':        'binary',
        'metric':          'auc',
        'boosting_type':   'gbdt',
        'num_leaves':      31,
        'learning_rate':   0.05,
        'feature_fraction': 0.9,
        'bagging_fraction':  0.8,
        'bagging_freq':       5,
        'scale_pos_weight':  (y_train == 0).sum() / max((y_train == 1).sum(), 1),
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    }

    bst = lgb.train(
        params, dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=[lgb.log_evaluation(period=50)],
    )

    elapsed = time.time() - t0
    print(f"Training time: {elapsed:.3f}s")
    return bst, X_test, y_test, elapsed

# ---------------------------------------------------------------------------
# 3. Evaluate
# ---------------------------------------------------------------------------
def evaluate(bst, X_test, y_test):
    y_proba = bst.predict(X_test)
    y_pred   = (y_proba > 0.5).astype(int)
    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    return auc, acc, f1, prec, rec

# ---------------------------------------------------------------------------
# 4. Inference benchmark
# ---------------------------------------------------------------------------
def inference_benchmark(bst, X_test):
    # Latency: 1 row
    t1 = time.time()
    for _ in range(1000):
        _ = bst.predict(X_test.iloc[[0]])
    latency_1 = (time.time() - t1) / 1000

    # Throughput: 1000 rows
    t1 = time.time()
    _ = bst.predict(X_test.iloc[:1000])
    latency_1000 = time.time() - t1

    return latency_1, latency_1000

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("LightGBM Benchmark — Credit Card Fraud Detection (CPU)")
    print("=" * 60)

    # Load
    df, load_time = load_data()

    # Train
    bst, X_test, y_test, train_time = train_model(df)

    # Evaluate
    auc, acc, f1, prec, rec = evaluate(bst, X_test, y_test)
    best_iter = bst.best_iteration

    # Inference
    latency_1, latency_1000 = inference_benchmark(bst, X_test)

    # Save model
    bst.save_model(MODEL_FILE)

    # Print results
    print(f"Best iteration:  {best_iter}")
    print(f"AUC-ROC:         {auc:.6f}")
    print(f"Accuracy:        {acc:.6f}")
    print(f"F1-Score:        {f1:.6f}")
    print(f"Precision:       {prec:.6f}")
    print(f"Recall:          {rec:.6f}")
    print(f"Inference latency (1 row):     {latency_1*1000:.3f} ms")
    print(f"Inference throughput (1000 rows): {latency_1000:.3f} s")
    print("=" * 60)

    # Save results JSON
    results = {
        "data_load_time_s":           round(load_time, 3),
        "training_time_s":            round(train_time, 3),
        "best_iteration":             best_iter,
        "auc_roc":                    round(auc, 6),
        "accuracy":                   round(acc, 6),
        "f1_score":                   round(f1, 6),
        "precision":                  round(prec, 6),
        "recall":                     round(rec, 6),
        "inference_latency_1_row_ms": round(latency_1 * 1000, 3),
        "inference_throughput_1000_rows_s": round(latency_1000, 3),
        "total_rows":                 int(df.shape[0]),
        "note": "synthetic data (real creditcard.csv not available)",
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {RESULTS_FILE}")

    # -------------------------------------------------------------------------
    # Flask API (runs after benchmark completes)
    # -------------------------------------------------------------------------
    print("Starting Flask API on 0.0.0.0:8000...")
    model_api = lgb.Booster(model_file=MODEL_FILE)
    feature_names = list(X_test.columns)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"}), 200

    @app.route("/predict", methods=["POST"])
    def predict():
        from flask import request
        data = request.get_json(force=True)
        if "features" not in data:
            return jsonify({"error": "missing 'features' field"}), 400
        features = np.array(data["features"]).reshape(1, -1)
        proba = float(model_api.predict(features)[0])
        return jsonify({"fraud_probability": round(proba, 6)})

    app.run(host="0.0.0.0", port=8000)