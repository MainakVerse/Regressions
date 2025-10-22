import numpy as np

class StackingRegressorScratch:
    def __init__(self, base_models, meta_model, holdout_ratio=0.2):
        """
        Stacking Regressor from scratch.
        
        Parameters:
            base_models (list): List of initialized base regressors (each must have fit/predict)
            meta_model (object): Meta-regressor (must have fit/predict)
            holdout_ratio (float): Fraction of data for meta-model training
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.holdout_ratio = holdout_ratio

    def fit(self, X, y):
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        n_samples = len(X)
        split = int((1 - self.holdout_ratio) * n_samples)

        # Split into base-training and meta-training sets
        X_base, y_base = X[:split], y[:split]
        X_meta, y_meta = X[split:], y[split:]

        # Train base models
        self.base_models_ = []
        meta_features = []

        for model in self.base_models:
            clone = model.__class__(**model.__dict__) if hasattr(model, '__dict__') else model
            clone.fit(X_base, y_base)
            preds = clone.predict(X_meta)
            meta_features.append(preds)
            self.base_models_.append(clone)

        # Create meta-feature matrix (n_meta_samples × n_base_models)
        meta_X = np.column_stack(meta_features)

        # Train meta-model on base model predictions
        self.meta_model.fit(meta_X, y_meta)
        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        meta_features = []

        for model in self.base_models_:
            meta_features.append(model.predict(X))

        meta_X = np.column_stack(meta_features)
        return self.meta_model.predict(meta_X)
