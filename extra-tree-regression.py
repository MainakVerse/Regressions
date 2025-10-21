import numpy as np

class ExtraTreeRegressorScratch(DecisionTreeRegressorScratch):
    """A single Extremely Randomized Tree."""
    def _best_split(self, X, y):
        best_feature, best_threshold, best_mse = None, None, float('inf')
        n_samples, n_features = X.shape

        # Randomized splitting instead of exhaustive search
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            if len(thresholds) == 0:
                continue

            # Pick random threshold instead of best one
            t = np.random.choice(thresholds)
            left_idx = X[:, feature] <= t
            right_idx = X[:, feature] > t

            if np.sum(left_idx) < self.min_samples_split or np.sum(right_idx) < self.min_samples_split:
                continue

            left_mse = self._mse(y[left_idx])
            right_mse = self._mse(y[right_idx])
            mse = (len(y[left_idx]) * left_mse + len(y[right_idx]) * right_mse) / len(y)

            if mse < best_mse:
                best_feature, best_threshold, best_mse = feature, t, mse

        return best_feature, best_threshold


class ExtraTreesRegressorScratch:
    """Ensemble of multiple Extra Trees."""
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2, sample_ratio=1.0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.sample_ratio = sample_ratio
        self.trees = []

    def fit(self, X, y):
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        n_samples = len(X)

        self.trees = []
        for _ in range(self.n_estimators):
            # Random subset of data (optional bootstrap)
            sample_idx = np.random.choice(n_samples, int(self.sample_ratio * n_samples), replace=True)
            X_sample, y_sample = X[sample_idx], y[sample_idx]

            tree = ExtraTreeRegressorScratch(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        return self

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(preds, axis=0)
