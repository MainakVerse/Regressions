import numpy as np

class AdaBoostRegressorScratch:
    def __init__(self, base_model_class, n_estimators=10, random_state=None, **base_model_params):
        """
        AdaBoost Regressor from scratch.

        Parameters:
            base_model_class (class): Base weak learner (e.g., DecisionTreeRegressorScratch)
            n_estimators (int): Number of boosting rounds
            random_state (int): Random seed
            base_model_params (dict): Parameters for the base model
        """
        self.base_model_class = base_model_class
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.base_model_params = base_model_params
        self.models = []
        self.model_weights = []

        if random_state:
            np.random.seed(random_state)

    def fit(self, X, y):
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        n_samples = len(y)
        weights = np.ones(n_samples) / n_samples

        self.models, self.model_weights = [], []

        for _ in range(self.n_estimators):
            # Weighted sampling
            sample_idx = np.random.choice(n_samples, n_samples, replace=True, p=weights)
            X_sample, y_sample = X[sample_idx], y[sample_idx]

            # Train weak learner
            model = self.base_model_class(**self.base_model_params)
            model.fit(X_sample, y_sample)

            # Compute weighted error
            y_pred = model.predict(X)
            err = np.sum(weights * np.abs(y - y_pred)) / np.sum(weights)
            err = np.clip(err, 1e-10, 0.999999)  # avoid div by zero

            # Compute model weight
            alpha = np.log((1 - err) / err)

            # Update sample weights
            weights *= np.exp(alpha * np.abs(y - y_pred))
            weights /= np.sum(weights)

            self.models.append(model)
            self.model_weights.append(alpha)

        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        preds = np.array([model.predict(X) for model in self.models])
        weighted_preds = np.average(preds, axis=0, weights=self.model_weights)
        return weighted_preds
