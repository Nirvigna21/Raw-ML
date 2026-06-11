import numpy as np


class LinearRegression:
    """
    Linear Regression using Gradient Descent.
    
    Math:
        y_pred = X @ w + b
        Loss   = MSE = (1/m) * sum((y_pred - y)^2)
        dw     = (2/m) * X.T @ (y_pred - y)
        db     = (2/m) * sum(y_pred - y)
    """

    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iters):
            y_pred = X @ self.weights + self.bias
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)

            dw = (2 / m) * X.T @ (y_pred - y)
            db = (2 / m) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict(self, X):
        return X @ self.weights + self.bias

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot  # R²

    def mse(self, X, y):
        return np.mean((self.predict(X) - y) ** 2)
