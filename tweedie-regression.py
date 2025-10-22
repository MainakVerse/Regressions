import numpy as np

class TweedieRegressorScratch:
    def __init__(self, power=1.5, max_iter=200, lr=0.01, tol=1e-6):
        """
        Tweedie Regressor (generalized exponential family model).

        Parameters:
            power (float): Tweedie power parameter (0=Normal, 1=Poisson, 2=Gamma)
            max_iter (int): Maximum iterations
            lr (float): Learning rate
            tol (float): Convergence tolerance
        """
        self.power = power
        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        self.beta = None

    def fit(self, X, y):
        """Fit Tweedie model using gradient descent."""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n_samples, n_features = X.shape

        X_b = np.c_[np.ones((n_samples, 1)), X]
        self.beta = np.zeros(X_b.shape[1])

        for _ in range(self.max_iter):
            mu = np.exp(X_b @ self.beta)
            # Gradient of negative log-likelihood (simplified form)
            grad = -X_b.T @ ((y - mu) * mu ** (1 - self.power))
            # Update step
            self.beta -= self.lr * grad

            if np.linalg.norm(grad) < self.tol:
                break

        return self

    def predict(self, X):
        """Predict mean μ = exp(Xβ)."""
        X = np.array(X, dtype=float)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return np.exp(X_b @ self.beta)
