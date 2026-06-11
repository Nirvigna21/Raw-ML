import numpy as np
from collections import Counter


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # leaf node class

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    """
    Decision Tree Classifier using Information Gain (Entropy) or Gini.

    Math (Information Gain):
        Entropy(S) = -sum(p_i * log2(p_i))
        IG(S, A)   = Entropy(S) - sum(|S_v|/|S| * Entropy(S_v))

    Math (Gini):
        Gini(S) = 1 - sum(p_i^2)
    """

    def __init__(self, max_depth=10, min_samples_split=2, criterion="gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None

    def fit(self, X, y):
        self.n_classes = len(set(y))
        self.root = self._grow(X, y, depth=0)
        return self

    def _impurity(self, y):
        counts = np.bincount(y)
        probs = counts / len(y)
        if self.criterion == "gini":
            return 1 - np.sum(probs ** 2)
        else:  # entropy
            probs = probs[probs > 0]
            return -np.sum(probs * np.log2(probs))

    def _info_gain(self, y, X_col, threshold):
        parent = self._impurity(y)
        left_mask = X_col <= threshold
        right_mask = ~left_mask
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return 0
        n = len(y)
        left_imp = self._impurity(y[left_mask])
        right_imp = self._impurity(y[right_mask])
        child = (left_mask.sum() / n) * left_imp + (right_mask.sum() / n) * right_imp
        return parent - child

    def _best_split(self, X, y):
        best_gain, best_feat, best_thresh = -1, None, None
        for feat in range(X.shape[1]):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                gain = self._info_gain(y, X[:, feat], thresh)
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, thresh
        return best_feat, best_thresh

    def _grow(self, X, y, depth):
        # stopping conditions
        if (depth >= self.max_depth or
                len(set(y)) == 1 or
                len(y) < self.min_samples_split):
            return Node(value=Counter(y).most_common(1)[0][0])

        feat, thresh = self._best_split(X, y)
        if feat is None:
            return Node(value=Counter(y).most_common(1)[0][0])

        left_mask = X[:, feat] <= thresh
        left = self._grow(X[left_mask], y[left_mask], depth + 1)
        right = self._grow(X[~left_mask], y[~left_mask], depth + 1)
        return Node(feature=feat, threshold=thresh, left=left, right=right)

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
