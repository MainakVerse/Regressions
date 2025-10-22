import numpy as np

class BlendingRegressorScratch:
    def __init__(self, base_models, meta_model, val_ratio=0.2):
        """
        Blending Regressor from scratch.

        Parameters:
            base_models (list): List of base regressors (must implement fit/predict)
            meta_model (object): Meta regressor (must implement fit/predict)
            val_ratio (float): Fraction of data for validation blending
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.val_ratio = val_ratio

    def fit(self, X, y):
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        n_samples = len(X)
        split = int((1 - self.val_ratio) * n_samples)

        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        self.base_models_ = []
        meta_features = []

        # Train base models
        for model in self.base_models:
            clone = model.__class__(**model.__dict__) if hasattr(model, '__dict__') else model
            clone.fit(X_train, y_train)
            self.base_models_.append(clone)

            # Get predictions for validation set
            preds = clone.predict(X_val)
            meta_features.append(preds)

        # Meta training data: each column = base model’s prediction
        meta_X = np.column_stack(meta_features)

        # Train meta-model on validation predictions
        self.meta_model.fit(meta_X, y_val)
        return self

    def predict(self, X):
        """Blend base model predictions through the meta-model."""
        X = np.array(X, dtype=float)
        meta_features = np.column_stack([m.predict(X) for m in self.base_models_])
        return self.meta_model.predict(meta_features)
