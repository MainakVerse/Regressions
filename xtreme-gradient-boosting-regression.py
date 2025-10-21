import numpy as np

class XGBoostRegressorScratch:
    def __init__(self, base_model_class, n_estimators=100, learning_rate=0.1,
                 max_depth=3, reg_lambda=1.0, reg_gamma=0.0, random_state=None, **base_model_params):
        """
        Simplified XGBoost Regressor implementation.
        """
        self.base_model_class = base_model_class
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda  # L2 regularization term
        self.reg_gamma = reg_gamma    # Tree complexity penalty
        self.random_state = random_state
        self.base_model_params = base_model_params
        self.models = []
        self.F0 = None

        if random_state:
            np.random.seed(random_state)

    def fit(self, X, y):
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        n_samples = len(y)
        self.F0 = np.mean(y)
        y_pred = np.full(n_samples, self.F0)
        self.models = []

        for _ in range(self.n_estimators):
            # 1️⃣ Compute gradient & hessian for squared loss
            grad = y_pred - y
            hess = np.ones_like(grad)

            # 2️⃣ Pseudo-residuals (negative gradient)
            residuals = -grad / (hess + self.reg_lambda)

            # 3️⃣ Fit weak learner to residuals
            tree = self.base_model_class(max_depth=self.max_depth, **self.base_model_params)
            tree.fit(X, residuals)

            # 4️⃣ Update predictions
            y_pred += self.learning_rate * tree.predict(X)
            self.models.append(tree)

        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        y_pred = np.full(X.shape[0], self.F0)
        for tree in self.models:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred
