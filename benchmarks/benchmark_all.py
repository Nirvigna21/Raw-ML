"""
benchmark_all.py
Runs all 6 from-scratch algorithms vs sklearn equivalents.
Results saved to results/benchmark_results.csv
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from sklearn.linear_model import LinearRegression as SkLinReg
from sklearn.linear_model import LogisticRegression as SkLogReg
from sklearn.tree import DecisionTreeClassifier as SkDT
from sklearn.cluster import KMeans as SkKMeans
from sklearn.svm import SVC as SkSVC
from sklearn.naive_bayes import GaussianNB as SkNB

from algorithms import (
    LinearRegression, LogisticRegression,
    DecisionTree, KMeans, SVM, GaussianNaiveBayes
)

scaler = StandardScaler()
results = []

# ── 1. Linear Regression ── Diabetes ────────────────────────────────────────
print("▶ Linear Regression...")
data = datasets.load_diabetes()
X, y = data.data, data.target
X = scaler.fit_transform(X)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

t0 = time.time()
our = LinearRegression(lr=0.05, n_iters=1000).fit(Xtr, ytr)
our_time = time.time() - t0
our_r2 = our.score(Xte, yte)

t0 = time.time()
sk = SkLinReg().fit(Xtr, ytr)
sk_time = time.time() - t0
sk_r2 = sk.score(Xte, yte)

results.append({"Algorithm": "Linear Regression", "Dataset": "Diabetes",
    "Ours_Score": round(our_r2, 4), "SK_Score": round(sk_r2, 4), "Metric": "R²",
    "Ours_Time": round(our_time, 4), "SK_Time": round(sk_time, 4)})
print(f"   R²  — Ours: {our_r2:.4f}  SK: {sk_r2:.4f}  | Time — Ours: {our_time:.3f}s  SK: {sk_time:.4f}s")

# ── 2. Logistic Regression ── Breast Cancer ──────────────────────────────────
print("▶ Logistic Regression...")
data = datasets.load_breast_cancer()
X = scaler.fit_transform(data.data)
Xtr, Xte, ytr, yte = train_test_split(X, data.target, test_size=0.2, random_state=42)

t0 = time.time()
our = LogisticRegression(lr=0.01, n_iters=1000).fit(Xtr, ytr)
our_time = time.time() - t0
our_acc = our.score(Xte, yte)

t0 = time.time()
sk = SkLogReg(max_iter=1000).fit(Xtr, ytr)
sk_time = time.time() - t0
sk_acc = sk.score(Xte, yte)

results.append({"Algorithm": "Logistic Regression", "Dataset": "Breast Cancer",
    "Ours_Score": round(our_acc, 4), "SK_Score": round(sk_acc, 4), "Metric": "Accuracy",
    "Ours_Time": round(our_time, 4), "SK_Time": round(sk_time, 4)})
print(f"   Acc — Ours: {our_acc:.4f}  SK: {sk_acc:.4f}  | Time — Ours: {our_time:.3f}s  SK: {sk_time:.4f}s")

# ── 3. Decision Tree ── Iris ─────────────────────────────────────────────────
print("▶ Decision Tree...")
data = datasets.load_iris()
Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

t0 = time.time()
our = DecisionTree(max_depth=4, criterion="gini").fit(Xtr, ytr)
our_time = time.time() - t0
our_acc = our.score(Xte, yte)

t0 = time.time()
sk = SkDT(max_depth=4, criterion="gini").fit(Xtr, ytr)
sk_time = time.time() - t0
sk_acc = sk.score(Xte, yte)

results.append({"Algorithm": "Decision Tree", "Dataset": "Iris",
    "Ours_Score": round(our_acc, 4), "SK_Score": round(sk_acc, 4), "Metric": "Accuracy",
    "Ours_Time": round(our_time, 4), "SK_Time": round(sk_time, 4)})
print(f"   Acc — Ours: {our_acc:.4f}  SK: {sk_acc:.4f}  | Time — Ours: {our_time:.3f}s  SK: {sk_time:.4f}s")

# ── 4. k-Means ── Digits ─────────────────────────────────────────────────────
print("▶ k-Means...")
data = datasets.load_digits()
X = scaler.fit_transform(data.data)

t0 = time.time()
our = KMeans(k=10, max_iters=100).fit(X)
our_time = time.time() - t0

t0 = time.time()
sk = SkKMeans(n_clusters=10, max_iter=100, random_state=42, n_init=1).fit(X)
sk_time = time.time() - t0

results.append({"Algorithm": "k-Means", "Dataset": "Digits",
    "Ours_Score": round(our.inertia_, 1), "SK_Score": round(sk.inertia_, 1),
    "Metric": "Inertia (lower=better)",
    "Ours_Time": round(our_time, 4), "SK_Time": round(sk_time, 4)})
print(f"   Inertia — Ours: {our.inertia_:.1f}  SK: {sk.inertia_:.1f}  | Time — Ours: {our_time:.3f}s  SK: {sk_time:.4f}s")

# ── 5. SVM ── Digits binary (0 vs 1) ─────────────────────────────────────────
print("▶ SVM...")
data = datasets.load_digits()
mask = data.target < 2
X = scaler.fit_transform(data.data[mask])
y = np.where(data.target[mask] == 0, -1, 1)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

t0 = time.time()
our = SVM(lr=0.001, lambda_param=0.01, n_iters=500).fit(Xtr, ytr)
our_time = time.time() - t0
our_acc = our.score(Xte, yte)

t0 = time.time()
sk = SkSVC(kernel="linear").fit(Xtr, ytr)
sk_time = time.time() - t0
sk_acc = sk.score(Xte, yte)

results.append({"Algorithm": "SVM", "Dataset": "Digits (0 vs 1)",
    "Ours_Score": round(our_acc, 4), "SK_Score": round(sk_acc, 4), "Metric": "Accuracy",
    "Ours_Time": round(our_time, 4), "SK_Time": round(sk_time, 4)})
print(f"   Acc — Ours: {our_acc:.4f}  SK: {sk_acc:.4f}  | Time — Ours: {our_time:.3f}s  SK: {sk_time:.4f}s")

# ── 6. Naive Bayes ── Wine ────────────────────────────────────────────────────
print("▶ Naive Bayes...")
data = datasets.load_wine()
X = scaler.fit_transform(data.data)
Xtr, Xte, ytr, yte = train_test_split(X, data.target, test_size=0.2, random_state=42)

t0 = time.time()
our = GaussianNaiveBayes().fit(Xtr, ytr)
our_time = time.time() - t0
our_acc = our.score(Xte, yte)

t0 = time.time()
sk = SkNB().fit(Xtr, ytr)
sk_time = time.time() - t0
sk_acc = sk.score(Xte, yte)

results.append({"Algorithm": "Naive Bayes", "Dataset": "Wine",
    "Ours_Score": round(our_acc, 4), "SK_Score": round(sk_acc, 4), "Metric": "Accuracy",
    "Ours_Time": round(our_time, 4), "SK_Time": round(sk_time, 4)})
print(f"   Acc — Ours: {our_acc:.4f}  SK: {sk_acc:.4f}  | Time — Ours: {our_time:.3f}s  SK: {sk_time:.4f}s")

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
df = pd.DataFrame(results)
df.to_csv("results/benchmark_results.csv", index=False)
print("\n✅ Results saved to results/benchmark_results.csv")
print(df[["Algorithm","Dataset","Metric","Ours_Score","SK_Score","Ours_Time","SK_Time"]].to_string(index=False))
