import numpy as np

class VotingRegressorScratch:
    def __init__(self, models, weights=None):
        """
        Voting Regressor from scratch.

        Parameters:
            models (list): List of initialized regressors (each must have fit/predict)
            weights (list): Optional list of weights for each model
        """
        self.models = models
        self.weights = np.array(weights) if weights is not None else None

    def fit(self, X, y):
        """Fit all base models."""
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        self.models_ = []
        for model in self.models:
            clone = model.__class__(**model.__dict__) if hasattr(model, '__dict__') else model
            clone.fit(X, y)
            self.models_.append(clone)
        return self

    def predict(self, X):
        """Predict by averaging model outputs."""
        X = np.array(X, dtype=float)
        preds = np.array([model.predict(X) for model in self.models_])

        if self.weights is None:
            return np.mean(preds, axis=0)
        else:
            weights = self.weights / np.sum(self.weights)
            return np.average(preds, axis=0, weights=weights)
