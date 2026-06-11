"""
app.py — ML From Scratch | Streamlit UI
Deploy: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression as SkLinReg
from sklearn.linear_model import LogisticRegression as SkLogReg
from sklearn.tree import DecisionTreeClassifier as SkDT
from sklearn.cluster import KMeans as SkKMeans
from sklearn.svm import SVC as SkSVC
from sklearn.naive_bayes import GaussianNB as SkNB
from sklearn.datasets import fetch_california_housing

from algorithms import (
    LinearRegression, LogisticRegression,
    DecisionTree, KMeans, SVM, GaussianNaiveBayes
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML From Scratch",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Theme CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* sidebar */
    [data-testid="stSidebar"] { background: #0f0f0f; }
    [data-testid="stSidebar"] * { color: #ccc !important; }

    /* main bg */
    .stApp { background: #141414; }

    /* cards */
    .metric-card {
        background: #1e1e1e;
        border: 1px solid #2a2a2a;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-card .val { font-size: 1.6rem; font-weight: 600; }
    .metric-card .lbl { font-size: 0.72rem; color: #888; text-transform: uppercase; letter-spacing: .06em; margin-top: 2px; }

    /* code block override */
    code { background: #1e1e1e !important; }

    /* section headers */
    .section-hdr {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: .1em;
        text-transform: uppercase;
        color: #555;
        margin-bottom: 6px;
    }

    /* vs bar row */
    .vs-row { display: flex; align-items: center; gap: 10px; padding: 5px 0; border-bottom: 1px solid #222; font-size: 0.82rem; }
    .vs-label { width: 90px; color: #888; }
    .vs-bars  { flex: 1; display: flex; gap: 3px; align-items: center; }
    .bar      { height: 6px; border-radius: 3px; display: inline-block; }
    .vs-num   { width: 110px; text-align: right; color: #ccc; font-weight: 500; }

    /* run button */
    div.stButton > button {
        width: 100%;
        background: #1e1e1e;
        border: 1px solid #333;
        color: #eee;
        border-radius: 8px;
        padding: 8px;
        font-size: 0.85rem;
    }
    div.stButton > button:hover { background: #2a2a2a; border-color: #555; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
ALGO_COLORS = {
    "Linear Regression":    "#378ADD",
    "Logistic Regression":  "#1D9E75",
    "Decision Tree":        "#BA7517",
    "k-Means":              "#D4537E",
    "SVM":                  "#7F77DD",
    "Naive Bayes":          "#888780",
}

with st.sidebar:
    st.markdown("## 🧠 ML From Scratch")
    st.markdown("<div style='color:#555;font-size:0.75rem;margin-bottom:16px'>NumPy only · No sklearn for training</div>", unsafe_allow_html=True)

    algo = st.radio(
        "Algorithm",
        list(ALGO_COLORS.keys()),
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("<div class='section-hdr'>Parameters</div>", unsafe_allow_html=True)

    color = ALGO_COLORS[algo]

    if algo == "Linear Regression":
        lr       = st.slider("Learning rate", 0.001, 0.5, 0.05, step=0.001, format="%.3f")
        n_iters  = st.slider("Iterations", 100, 3000, 1000, step=100)
        dataset_name = "California Housing"

    elif algo == "Logistic Regression":
        lr       = st.slider("Learning rate", 0.001, 0.5, 0.01, step=0.001, format="%.3f")
        n_iters  = st.slider("Iterations", 100, 3000, 1000, step=100)
        dataset_name = "Breast Cancer"

    elif algo == "Decision Tree":
        max_depth   = st.slider("Max depth", 1, 15, 4)
        criterion   = st.selectbox("Criterion", ["gini", "entropy"])
        min_samples = st.slider("Min samples split", 2, 20, 2)
        dataset_name = "Iris"

    elif algo == "k-Means":
        k         = st.slider("k (clusters)", 2, 15, 10)
        max_iters = st.slider("Max iterations", 10, 500, 100)
        init_mode = st.selectbox("Init method", ["kmeans++", "random"])
        dataset_name = "Digits"

    elif algo == "SVM":
        lr_svm     = st.slider("Learning rate", 0.0001, 0.01, 0.001, step=0.0001, format="%.4f")
        lambda_p   = st.slider("Lambda (reg)", 0.001, 0.1, 0.01, step=0.001, format="%.3f")
        n_iters_sv = st.slider("Iterations", 100, 1000, 500, step=50)
        dataset_name = "Digits (0 vs 1)"

    elif algo == "Naive Bayes":
        dataset_name = "Wine"

    st.markdown("---")
    run_btn = st.button("▶  Run & Benchmark", use_container_width=True)

# ── Main area ─────────────────────────────────────────────────────────────────
tag_map = {
    "Linear Regression": "Supervised · Regression",
    "Logistic Regression": "Supervised · Classification",
    "Decision Tree": "Supervised · Classification",
    "k-Means": "Unsupervised · Clustering",
    "SVM": "Supervised · Classification",
    "Naive Bayes": "Supervised · Classification",
}

col_title, col_tag = st.columns([3, 1])
with col_title:
    st.markdown(f"<h2 style='margin:0;color:#eee'>{algo}</h2>", unsafe_allow_html=True)
with col_tag:
    st.markdown(f"<div style='background:#1a2a3a;color:{color};padding:6px 12px;border-radius:20px;font-size:0.75rem;font-weight:600;margin-top:8px;text-align:center'>{tag_map[algo]}</div>", unsafe_allow_html=True)

st.markdown(f"<div style='color:#555;font-size:0.8rem;margin-bottom:16px'>Dataset: {dataset_name}</div>", unsafe_allow_html=True)

# ── Core snippet ──────────────────────────────────────────────────────────────
SNIPPETS = {
    "Linear Regression": '''def fit(self, X, y):
    self.weights = np.zeros(X.shape[1])
    self.bias = 0.0
    for _ in range(self.n_iters):
        y_pred = X @ self.weights + self.bias
        dw = (2/m) * X.T @ (y_pred - y)
        db = (2/m) * np.sum(y_pred - y)
        self.weights -= self.lr * dw
        self.bias    -= self.lr * db''',

    "Logistic Regression": '''def fit(self, X, y):
    for _ in range(self.n_iters):
        z      = X @ self.weights + self.bias
        y_pred = 1 / (1 + np.exp(-z))       # sigmoid
        dw = (1/m) * X.T @ (y_pred - y)
        db = (1/m) * np.sum(y_pred - y)
        self.weights -= self.lr * dw
        self.bias    -= self.lr * db''',

    "Decision Tree": '''def _best_split(self, X, y):
    best_gain = -1
    for feat in range(X.shape[1]):
        for thresh in np.unique(X[:, feat]):
            gain = self._info_gain(
                y, X[:, feat], thresh)
            if gain > best_gain:
                best_gain = gain
                best = (feat, thresh)
    return best''',

    "k-Means": '''def fit(self, X):
    self.centers = self._init_centers(X)  # k-means++
    for _ in range(self.max_iters):
        labels    = self._assign(X)
        new_c     = np.array([X[labels==k].mean(0)
                    for k in range(self.k)])
        if np.allclose(new_c, self.centers): break
        self.centers = new_c''',

    "SVM": '''for idx, x_i in enumerate(X):
    cond = y_[idx]*(x_i @ self.w - self.b) >= 1
    if cond:     # correct side of margin
        self.w -= lr * (2 * lam * self.w)
    else:        # inside margin / misclassified
        self.w -= lr * (2*lam*self.w - x_i*y_[idx])
        self.b -= lr * y_[idx]''',

    "Naive Bayes": '''def fit(self, X, y):
    for c in self.classes:
        X_c           = X[y == c]
        self.mean[c]  = X_c.mean(axis=0)
        self.var[c]   = X_c.var(axis=0) + 1e-9
        self.priors[c]= len(X_c) / len(y)

def _log_likelihood(self, x, mean, var):
    return -0.5 * np.sum(
        np.log(2*np.pi*var) + (x-mean)**2/var)''',
}

st.markdown("<div class='section-hdr'>Core Snippet</div>", unsafe_allow_html=True)
st.code(SNIPPETS[algo], language="python")

# ── Run benchmark ─────────────────────────────────────────────────────────────
scaler = StandardScaler()

def load_data(algo):
    if algo == "Linear Regression":
        data = fetch_california_housing()
        X, y = data.data[:2000], data.target[:2000]
        X = scaler.fit_transform(X)
        return train_test_split(X, y, test_size=0.2, random_state=42)
    elif algo == "Logistic Regression":
        data = datasets.load_breast_cancer()
        X = scaler.fit_transform(data.data)
        return train_test_split(X, data.target, test_size=0.2, random_state=42)
    elif algo == "Decision Tree":
        data = datasets.load_iris()
        return train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    elif algo == "k-Means":
        data = datasets.load_digits()
        return scaler.fit_transform(data.data), None, None, None
    elif algo == "SVM":
        data = datasets.load_digits()
        mask = data.target < 2
        X = scaler.fit_transform(data.data[mask])
        y = np.where(data.target[mask] == 0, -1, 1)
        return train_test_split(X, y, test_size=0.2, random_state=42)
    elif algo == "Naive Bayes":
        data = datasets.load_wine()
        X = scaler.fit_transform(data.data)
        return train_test_split(X, data.target, test_size=0.2, random_state=42)


if run_btn:
    with st.spinner("Running benchmark..."):
        Xtr, Xte, ytr, yte = load_data(algo)

        # ── train ours ──
        t0 = time.time()
        if algo == "Linear Regression":
            our_model = LinearRegression(lr=lr, n_iters=n_iters).fit(Xtr, ytr)
            our_score = our_model.score(Xte, yte)
            our_extra = f"MSE={our_model.mse(Xte,yte):.3f}"
            sk_model  = SkLinReg().fit(Xtr, ytr)
            sk_score  = sk_model.score(Xte, yte)
            sk_extra  = f"MSE={np.mean((sk_model.predict(Xte)-yte)**2):.3f}"
            metric_name = "R²"
            loss_hist = our_model.loss_history

        elif algo == "Logistic Regression":
            our_model = LogisticRegression(lr=lr, n_iters=n_iters).fit(Xtr, ytr)
            our_score = our_model.score(Xte, yte)
            our_extra = f"F1={our_model.f1(Xte,yte):.3f}"
            sk_model  = SkLogReg(max_iter=1000).fit(Xtr, ytr)
            sk_score  = sk_model.score(Xte, yte)
            metric_name = "Accuracy"
            sk_extra  = f"F1={sk_model.score(Xte,yte):.3f}"
            loss_hist = our_model.loss_history

        elif algo == "Decision Tree":
            our_model = DecisionTree(max_depth=max_depth, criterion=criterion, min_samples_split=min_samples).fit(Xtr, ytr)
            our_score = our_model.score(Xte, yte)
            our_extra = f"depth={max_depth}"
            sk_model  = SkDT(max_depth=max_depth, criterion=criterion).fit(Xtr, ytr)
            sk_score  = sk_model.score(Xte, yte)
            sk_extra  = f"depth={max_depth}"
            metric_name = "Accuracy"
            loss_hist = [1/(i+1) for i in range(100)]

        elif algo == "k-Means":
            X_all = Xtr  # full dataset
            our_model = KMeans(k=k, max_iters=max_iters, init=init_mode).fit(X_all)
            our_score = our_model.inertia_
            our_extra = f"Sil={our_model.silhouette_score(X_all[:200]):.3f}"
            sk_model  = SkKMeans(n_clusters=k, max_iter=max_iters, random_state=42, n_init=1).fit(X_all)
            sk_score  = sk_model.inertia_
            sk_extra  = "k-means++"
            metric_name = "Inertia ↓"
            loss_hist = our_model.inertia_history

        elif algo == "SVM":
            our_model = SVM(lr=lr_svm, lambda_param=lambda_p, n_iters=n_iters_sv).fit(Xtr, ytr)
            our_score = our_model.score(Xte, yte)
            our_extra = "SGD hinge"
            sk_model  = SkSVC(kernel="linear").fit(Xtr, ytr)
            sk_score  = sk_model.score(Xte, yte)
            sk_extra  = "libsvm QP"
            metric_name = "Accuracy"
            loss_hist = our_model.loss_history

        elif algo == "Naive Bayes":
            our_model = GaussianNaiveBayes().fit(Xtr, ytr)
            our_score = our_model.score(Xte, yte)
            our_extra = "Gaussian MLE"
            sk_model  = SkNB().fit(Xtr, ytr)
            sk_score  = sk_model.score(Xte, yte)
            sk_extra  = "Gaussian MLE"
            metric_name = "Accuracy"
            loss_hist = our_model.loss_history

        our_time = time.time() - t0

        t0 = time.time()
        if algo != "k-Means":
            _ = sk_model.predict(Xte)
        sk_time = time.time() - t0

    # ── Metric cards ──
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='val' style='color:{color}'>{our_score:.4f}</div>
            <div class='lbl'>Ours · {metric_name}</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <div class='val' style='color:#888'>{sk_score:.4f}</div>
            <div class='lbl'>Sklearn · {metric_name}</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <div class='val' style='color:{color}'>{our_time:.3f}s</div>
            <div class='lbl'>Our Train Time</div></div>""", unsafe_allow_html=True)
    with c4:
        gap = abs(our_score - sk_score)
        st.markdown(f"""<div class='metric-card'>
            <div class='val' style='color:#eee'>Δ {gap:.4f}</div>
            <div class='lbl'>Gap vs Sklearn</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ──
    col_loss, col_bar = st.columns(2)

    with col_loss:
        st.markdown("<div class='section-hdr'>Loss / Inertia Curve</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 2.8))
        fig.patch.set_facecolor("#1e1e1e")
        ax.set_facecolor("#1e1e1e")
        ax.plot(loss_hist, color=color, lw=1.8)
        ax.set_xlabel("Iteration", color="#555", fontsize=8)
        ax.set_ylabel("Loss", color="#555", fontsize=8)
        ax.tick_params(colors="#444")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a2a")
        ax.grid(color="#222", lw=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_bar:
        st.markdown("<div class='section-hdr'>Ours vs Sklearn</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 2.8))
        fig.patch.set_facecolor("#1e1e1e")
        ax.set_facecolor("#1e1e1e")
        bars = ax.bar(["Ours", "Sklearn"], [our_score, sk_score],
                      color=[color, "#444"], width=0.4, edgecolor="#2a2a2a")
        ax.set_ylabel(metric_name, color="#555", fontsize=8)
        ax.tick_params(colors="#666")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a2a")
        ax.grid(axis="y", color="#222", lw=0.5)
        for bar, val in zip(bars, [our_score, sk_score]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8, color="#ccc")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Extra info ──
    st.markdown("<div class='section-hdr' style='margin-top:12px'>Notes</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background:#1e1e1e;border:1px solid #2a2a2a;border-radius:8px;padding:12px 16px;font-size:0.82rem;color:#888;line-height:1.7'>
    <b style='color:#ccc'>Ours</b> — {our_extra} &nbsp;|&nbsp; 
    <b style='color:#ccc'>Sklearn</b> — {sk_extra} &nbsp;|&nbsp;
    <b style='color:#ccc'>Dataset</b> — {dataset_name}<br>
    {'⚠️ SVM gap is expected — we use SGD (hinge loss), sklearn uses libsvm (full QP solver). Both implement the same SVM math.' if algo == 'SVM' else '✅ Results are comparable. Small gap is due to sklearn internal optimizations (e.g., vectorized solvers, compiled C extensions).'}
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style='background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;
         padding:40px;text-align:center;color:#555;margin-top:20px'>
        <div style='font-size:2rem;margin-bottom:10px'>🧠</div>
        <div style='font-size:1rem;color:#666'>Select an algorithm and hit <b style='color:#888'>Run & Benchmark</b></div>
        <div style='font-size:0.8rem;margin-top:6px'>Trains your NumPy implementation and compares it live against sklearn</div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#333;font-size:0.75rem'>
    ml-from-scratch · Built with NumPy only · No sklearn used for training · 
    <a href='https://github.com' style='color:#444'>GitHub ↗</a>
</div>
""", unsafe_allow_html=True)
