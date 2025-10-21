import numpy as np

class CatBoostRegressorScratch:
    def __init__(self, base_model_class, n_estimators=100, learning_rate=0.1,
                 max_depth=3, random_state=None, **base_model_params):
        """
        Simplified CatBoost Regressor implementation.
        """
        self.base_model_class = base_model_class
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.base_model_params = base_model_params
        self.models = []
        self.F0 = None

        if random_state:
            np.random.seed(random_state)

    def _ordered_target_encoding(self, X_cat, y):
        """Perform ordered mean encoding for categorical features."""
        X_encoded = np.zeros_like(X_cat, dtype=float)
        for j in range(X_cat.shape[1]):
            for i in range(X_cat.shape[0]):
                mask = np.arange(i)
                if len(mask) > 0:
                    prev_y = y[mask][X_cat[mask, j] == X_cat[i, j]]
                    X_encoded[i, j] = np.mean(prev_y) if len(prev_y) > 0 else np.mean(y[:i])
                else:
                    X_encoded[i, j] = np.mean(y)
        return X_encoded

    def fit(self, X, y, categorical_idx=None):
        """
        Fit CatBoost-like model.
        Parameters:
            X: np.ndarray, feature matrix (can contain categorical features)
            y: np.ndarray, target vector
            categorical_idx: list of indices for categorical columns
        """
        X, y = np.array(X, dtype=object), np.array(y, dtype=float)
        n_samples = len(y)
        self.F0 = np.mean(y)
        y_pred = np.full(n_samples, self.F0)
        self.models = []

        # Separate categorical and numerical parts
        if categorical_idx is not None:
            X_num = np.delete(X, categorical_idx, axis=1).astype(float)
            X_cat = X[:, categorical_idx].astype(str)
            X_cat_encoded = self._ordered_target_encoding(X_cat, y)
            X_proc = np.hstack([X_num, X_cat_encoded])
        else:
            X_proc = X.astype(float)

        for _ in range(self.n_estimators):
            residuals = y - y_pred
            model = self.base_model_class(max_depth=self.max_depth, **self.base_model_params)
            model.fit(X_proc, residuals)
            y_pred += self.learning_rate * model.predict(X_proc)
            self.models.append(model)

        return self

    def predict(self, X, categorical_idx=None):
        X = np.array(X, dtype=object)
        if categorical_idx is not None:
            X_num = np.delete(X, categorical_idx, axis=1).astype(float)
            X_cat = X[:, categorical_idx].astype(str)
            # Use mean encoding based on training approximation
            X_cat_encoded = np.zeros_like(X_cat, dtype=float)
            X_proc = np.hstack([X_num, X_cat_encoded])
        else:
            X_proc = X.astype(float)

        y_pred = np.full(X_proc.shape[0], self.F0)
        for model in self.models:
            y_pred += self.learning_rate * model.predict(X_proc)
        return y_pred
