import numpy as np

class PoissonRegressionScratch:
    def __init__(self, max_iter=100, lr=0.01, tol=1e-6):
        """
        Poisson Regression from scratch.

        Parameters:
            max_iter (int): Maximum number of iterations
            lr (float): Learning rate for gradient descent
            tol (float): Convergence tolerance
        """
        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        self.beta = None

    def fit(self, X, y):
        """Fit Poisson regression model."""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n_samples, n_features = X.shape

        # Add intercept
        X_b = np.c_[np.ones((n_samples, 1)), X]
        self.beta = np.zeros(X_b.shape[1])

        for _ in range(self.max_iter):
            # Predicted mean λ = exp(Xβ)
            lambda_ = np.exp(X_b @ self.beta)
            # Gradient of log-likelihood
            grad = X_b.T @ (y - lambda_)
            # Hessian (for Newton-Raphson update)
            H = -X_b.T @ (lambda_[:, None] * X_b)
            # Update rule: β ← β - H⁻¹g
            try:
                delta = np.linalg.inv(H) @ grad
            except np.linalg.LinAlgError:
                delta = grad * self.lr  # fallback: simple gradient step

            self.beta -= delta

            if np.linalg.norm(delta) < self.tol:
                break

        return self

    def predict(self, X):
        """Predict expected count values."""
        X = np.array(X, dtype=float)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return np.exp(X_b @ self.beta)
