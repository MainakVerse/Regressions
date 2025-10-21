import numpy as np

class LightGBMRegressorScratch:
    def __init__(self, base_model_class, n_estimators=100, learning_rate=0.1, 
                 max_depth=3, top_rate=0.2, other_rate=0.1, random_state=None, **base_model_params):
        """
        LightGBM-like regressor (simplified GOSS version).
        """
        self.base_model_class = base_model_class
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.top_rate = top_rate      # keep fraction of large gradient samples
        self.other_rate = other_rate  # random fraction of small gradient samples
        self.random_state = random_state
        self.base_model_params = base_model_params
        self.models = []
        self.F0 = None

        if random_state:
            np.random.seed(random_state)

    def _goss_sampling(self, X, grad):
        """Gradient-based One-Side Sampling (GOSS)."""
        n_samples = len(grad)
        abs_grad = np.abs(grad)
        sorted_idx = np.argsort(-abs_grad)

        top_n = int(self.top_rate * n_samples)
        other_n = int(self.other_rate * n_samples)

        large_idx = sorted_idx[:top_n]
        small_idx = np.random.choice(sorted_idx[top_n:], other_n, replace=False)
        sampled_idx = np.concatenate([large_idx, small_idx])

        # Scale weights for unbiased estimation
        weights = np.ones_like(grad)
        weights[small_idx] *= (1 - self.top_rate) / self.other_rate

        return X[sampled_idx], grad[sampled_idx], weights[sampled_idx]

    def fit(self, X, y):
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        n_samples = len(y)
        self.F0 = np.mean(y)
        y_pred = np.full(n_samples, self.F0)
        self.models = []

        for _ in range(self.n_estimators):
            # 1️⃣ Compute gradients for squared loss
            grad = y_pred - y

            # 2️⃣ GOSS sampling
            X_sample, grad_sample, weights = self._goss_sampling(X, grad)

            # 3️⃣ Fit weak learner on sampled residuals
            model = self.base_model_class(max_depth=self.max_depth, **self.base_model_params)
            model.fit(X_sample, -grad_sample * weights)

            # 4️⃣ Update predictions
            y_pred += self.learning_rate * model.predict(X)
            self.models.append(model)

        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        y_pred = np.full(X.shape[0], self.F0)
        for model in self.models:
            y_pred += self.learning_rate * model.predict(X)
        return y_pred
