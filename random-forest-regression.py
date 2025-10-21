import numpy as np

class RandomForestRegressorScratch:
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2,
                 sample_ratio=0.8, feature_ratio=0.8):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.sample_ratio = sample_ratio
        self.feature_ratio = feature_ratio
        self.trees = []
        self.features_per_tree = []

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n_samples, n_features = X.shape

        for _ in range(self.n_estimators):
            # Random sample of data
            sample_idx = np.random.choice(n_samples, int(self.sample_ratio * n_samples), replace=True)
            X_sample, y_sample = X[sample_idx], y[sample_idx]

            # Random subset of features
            feature_idx = np.random.choice(n_features, int(self.feature_ratio * n_features), replace=False)
            self.features_per_tree.append(feature_idx)

            # Train tree on selected features
            tree = DecisionTreeRegressorScratch(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample[:, feature_idx], y_sample)
            self.trees.append(tree)
        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        preds = []
        for tree, f_idx in zip(self.trees, self.features_per_tree):
            preds.append(tree.predict(X[:, f_idx]))
        return np.mean(preds, axis=0)
