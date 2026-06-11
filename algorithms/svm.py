import numpy as np


class SVM:
    """
    Support Vector Machine (Soft-Margin) using SGD.

    Math:
        Objective : min (1/2)||w||^2 + C * sum(max(0, 1 - y_i*(w·x_i - b)))
        Hinge loss: L = max(0, 1 - y*(w·x - b))

        Gradient update:
            if y*(w·x - b) >= 1  (correctly classified):
                dw = 2 * lambda * w
                db = 0
            else (inside margin or misclassified):
                dw = 2 * lambda * w - y*x
                db = -y

    Note: We implement the SGD version (not full QP solver).
          sklearn uses libsvm (QP) so speed/accuracy will differ — this is expected and documented.
    """

    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        m, n = X.shape
        y_ = np.where(y <= 0, -1, 1)  # ensure labels are -1/+1
        self.weights = np.zeros(n)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iters):
            hinge_total = 0
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - x_i * y_[idx])
                    self.bias -= self.lr * y_[idx]
                    hinge_total += 1 - y_[idx] * (np.dot(x_i, self.weights) - self.bias)

            loss = self.lambda_param * np.dot(self.weights, self.weights) + hinge_total / m
            self.loss_history.append(loss)

        return self

    def predict(self, X):
        return np.sign(X @ self.weights - self.bias).astype(int)

    def score(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        return np.mean(self.predict(X) == y_)
