import numpy as np

class AveragingRegressorScratch:
    def __init__(self, models, dynamic_weights=True, epsilon=1e-8):
        """
        Averaging Regressor (custom ensemble).

        Parameters:
            models (list): List of initialized regressors (each must have fit/predict)
            dynamic_weights (bool): If True, weights are based on validation performance
            epsilon (float): Small constant to avoid division by zero
        """
        self.models = models
        self.dynamic_weights = dynamic_weights
        self.epsilon = epsilon
        self.weights = None

    def fit(self, X, y, val_split=0.2):
        """Fit models and optionally compute dynamic weights."""
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        n = len(X)
        split = int((1 - val_split) * n)

        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        self.models_ = []
        errors = []

        for model in self.models:
            clone = model.__class__(**model.__dict__) if hasattr(model, '__dict__') else model
            clone.fit(X_train, y_train)
            self.models_.append(clone)

            if self.dynamic_weights:
                preds = clone.predict(X_val)
                mse = np.mean((y_val - preds) ** 2)
                errors.append(mse)

        # Compute dynamic weights inversely proportional to validation MSE
        if self.dynamic_weights:
            inv_errors = 1 / (np.array(errors) + self.epsilon)
            self.weights = inv_errors / np.sum(inv_errors)
        else:
            self.weights = np.ones(len(self.models)) / len(self.models)

        return self

    def predict(self, X):
        """Predict by averaging weighted model predictions."""
        X = np.array(X, dtype=float)
        preds = np.array([m.predict(X) for m in self.models_])
        return np.average(preds, axis=0, weights=self.weights)
