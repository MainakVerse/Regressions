import numpy as np

class GradientBoostingRegressorScratch:
    def __init__(self, base_model_class, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None, **base_model_params):
        """
        Gradient Boosting Regressor from scratch.

        Parameters:
            base_model_class (class): Weak learner class (e.g., DecisionTreeRegressorScratch)
            n_estimators (int): Number of boosting stages
            learning_rate (float): Shrinkage parameter
            max_depth (int): Depth of weak learners
            random_state (int): Random seed
            base_model_params (dict): Additional base model parameters
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

    def fit(self, X, y):
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        n_samples = len(y)

        # Initial prediction = mean of y
        self.F0 = np.mean(y)
        y_pred = np.full(n_samples, self.F0)

        self.models = []

        for _ in range(self.n_estimators):
            # Compute residuals (negative gradient)
            residuals = y - y_pred

            # Fit weak learner to residuals
            model = self.base_model_class(max_depth=self.max_depth, **self.base_model_params)
            model.fit(X, residuals)

            # Update stage prediction
            y_pred += self.learning_rate * model.predict(X)
            self.models.append(model)

        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        y_pred = np.full(X.shape[0], self.F0)
        for model in self.models:
            y_pred += self.learning_rate * model.predict(X)
        return y_pred
