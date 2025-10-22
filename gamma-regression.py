import numpy as np

class GammaRegressionScratch:
    def __init__(self, max_iter=100, lr=0.01, tol=1e-6):
        """
        Gamma Regression from scratch (log link).
        
        Parameters:
            max_iter (int): Max iterations
            lr (float): Learning rate for gradient descent
            tol (float): Convergence tolerance
        """
        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        self.beta = None

    def fit(self, X, y):
        """Fit Gamma regression model."""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n_samples, n_features = X.shape

        # Add intercept
        X_b = np.c_[np.ones((n_samples, 1)), X]
        self.beta = np.zeros(X_b.shape[1])

        for _ in range(self.max_iter):
            # Mean prediction
            mu = np.exp(X_b @ self.beta)
            # Gradient and Hessian (from log-likelihood)
            grad = X_b.T @ ((y - mu) / (mu ** 2))
            H = -X_b.T @ np.diag(2 * (y / (mu ** 3))) @ X_b

            try:
                delta = np.linalg.inv(H) @ grad
            except np.linalg.LinAlgError:
                delta = grad * self.lr  # fallback

            self.beta -= delta

            if np.linalg.norm(delta) < self.tol:
                break

        return self

    def predict(self, X):
        """Predict mean response μ = exp(Xβ)."""
        X = np.array(X, dtype=float)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return np.exp(X_b @ self.beta)
