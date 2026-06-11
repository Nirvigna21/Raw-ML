import numpy as np


class LogisticRegression:
    """
    Binary Logistic Regression using Gradient Descent.

    Math:
        z      = X @ w + b
        y_pred = sigmoid(z) = 1 / (1 + exp(-z))
        Loss   = BCE = -(1/m) * sum(y*log(p) + (1-y)*log(1-p))
        dw     = (1/m) * X.T @ (y_pred - y)
        db     = (1/m) * sum(y_pred - y)
    """

    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iters):
            z = X @ self.weights + self.bias
            y_pred = self._sigmoid(z)

            eps = 1e-9
            loss = -np.mean(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))
            self.loss_history.append(loss)

            dw = (1 / m) * X.T @ (y_pred - y)
            db = (1 / m) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def f1(self, X, y):
        y_pred = self.predict(X)
        tp = np.sum((y_pred == 1) & (y == 1))
        fp = np.sum((y_pred == 1) & (y == 0))
        fn = np.sum((y_pred == 0) & (y == 1))
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        return 2 * precision * recall / (precision + recall + 1e-9)
