import numpy as np

class HistGradientBoostingRegressorScratch:
    def __init__(self, base_model_class, n_estimators=100, learning_rate=0.1,
                 max_depth=3, n_bins=32, random_state=None, **base_model_params):
        """
        Histogram-Based Gradient Boosting Regressor (simplified).
        """
        self.base_model_class = base_model_class
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.random_state = random_state
        self.base_model_params = base_model_params
        self.models = []
        self.bin_edges = None
        self.F0 = None

        if random_state:
            np.random.seed(random_state)

    def _bin_data(self, X):
        """Discretize features into bins."""
        n_samples, n_features = X.shape
        X_binned = np.zeros_like(X, dtype=int)
        self.bin_edges = []

        for j in range(n_features):
            edges = np.linspace(np.min(X[:, j]), np.max(X[:, j]), self.n_bins + 1)
            self.bin_edges.append(edges)
            X_binned[:, j] = np.digitize(X[:, j], edges) - 1
        return X_binned

    def fit(self, X, y):
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        n_samples = len(y)

        # 1️⃣ Initial prediction = mean of y
        self.F0 = np.mean(y)
        y_pred = np.full(n_samples, self.F0)

        # 2️⃣ Bin data
        X_binned = self._bin_data(X)

        self.models = []
        for _ in range(self.n_estimators):
            # Compute pseudo-residuals
            residuals = y - y_pred

            # Replace binned X for fast split finding
            model = self.base_model_class(max_depth=self.max_depth, **self.base_model_params)
            model.fit(X_binned, residuals)

            # Update predictions
            y_pred += self.learning_rate * model.predict(X_binned)
            self.models.append(model)

        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        X_binned = np.zeros_like(X, dtype=int)

        # Digitize new data using training bin edges
        for j, edges in enumerate(self.bin_edges):
            X_binned[:, j] = np.digitize(X[:, j], edges) - 1

        y_pred = np.full(X_binned.shape[0], self.F0)
        for model in self.models:
            y_pred += self.learning_rate * model.predict(X_binned)
        return y_pred
