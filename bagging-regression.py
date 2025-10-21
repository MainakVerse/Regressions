import numpy as np

class BaggingRegressorScratch:
    def __init__(self, base_model_class, n_estimators=10, sample_ratio=0.8, random_state=None, **base_model_params):
        """
        Bagging Regressor from scratch.

        Parameters:
            base_model_class (class): Class of the base model (e.g. DecisionTreeRegressorScratch)
            n_estimators (int): Number of base learners
            sample_ratio (float): Fraction of samples for each bootstrap dataset
            random_state (int): Random seed
            base_model_params (dict): Parameters for the base model
        """
        self.base_model_class = base_model_class
        self.n_estimators = n_estimators
        self.sample_ratio = sample_ratio
        self.random_state = random_state
        self.base_model_params = base_model_params
        self.models = []

        if random_state:
            np.random.seed(random_state)

    def fit(self, X, y):
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        n_samples = len(X)

        self.models = []
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            sample_idx = np.random.choice(n_samples, int(self.sample_ratio * n_samples), replace=True)
            X_sample, y_sample = X[sample_idx], y[sample_idx]

            # Train a new model
            model = self.base_model_class(**self.base_model_params)
            model.fit(X_sample, y_sample)
            self.models.append(model)
        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0)
