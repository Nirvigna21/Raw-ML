import numpy as np


class KMeans:
    """
    k-Means Clustering with k-means++ initialization.

    Math:
        Assignment : label_i = argmin_k ||x_i - c_k||^2
        Update     : c_k = mean(x_i for x_i in cluster k)
        Inertia    : sum over all points of ||x_i - c_{label_i}||^2
    """

    def __init__(self, k=3, max_iters=300, tol=1e-4, init="kmeans++"):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.init = init
        self.centers = None
        self.labels_ = None
        self.inertia_ = None
        self.inertia_history = []

    def _init_centers(self, X):
        if self.init == "random":
            idx = np.random.choice(len(X), self.k, replace=False)
            return X[idx].copy()

        # k-means++ init
        centers = [X[np.random.randint(len(X))]]
        for _ in range(1, self.k):
            dists = np.array([min(np.linalg.norm(x - c) ** 2 for c in centers) for x in X])
            probs = dists / dists.sum()
            centers.append(X[np.random.choice(len(X), p=probs)])
        return np.array(centers)

    def _assign(self, X):
        dists = np.linalg.norm(X[:, None] - self.centers[None], axis=2)  # (m, k)
        return np.argmin(dists, axis=1)

    def _inertia(self, X, labels):
        total = 0
        for k in range(self.k):
            mask = labels == k
            if mask.any():
                total += np.sum(np.linalg.norm(X[mask] - self.centers[k], axis=1) ** 2)
        return total

    def fit(self, X):
        self.centers = self._init_centers(X)
        self.inertia_history = []

        for _ in range(self.max_iters):
            labels = self._assign(X)
            new_centers = np.array([
                X[labels == k].mean(axis=0) if (labels == k).any() else self.centers[k]
                for k in range(self.k)
            ])
            self.inertia_history.append(self._inertia(X, labels))

            if np.linalg.norm(new_centers - self.centers) < self.tol:
                break
            self.centers = new_centers

        self.labels_ = self._assign(X)
        self.inertia_ = self._inertia(X, self.labels_)
        return self

    def predict(self, X):
        return self._assign(X)

    def silhouette_score(self, X):
        labels = self.labels_
        scores = []
        for i, x in enumerate(X):
            same = X[labels == labels[i]]
            a = np.mean(np.linalg.norm(same - x, axis=1)) if len(same) > 1 else 0
            b_vals = [
                np.mean(np.linalg.norm(X[labels == k] - x, axis=1))
                for k in range(self.k) if k != labels[i] and (labels == k).any()
            ]
            b = min(b_vals) if b_vals else 0
            scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0)
        return np.mean(scores)
