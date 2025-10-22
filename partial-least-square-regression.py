import numpy as np

class PartialLeastSquaresRegressionScratch:
    def __init__(self, n_components=2):
        """
        Partial Least Squares Regression (PLS) from scratch.

        Parameters:
            n_components (int): Number of latent components to extract.
        """
        self.n_components = n_components
        self.W = None
        self.P = None
        self.Q = None
        self.mean_X = None
        self.mean_y = None

    def fit(self, X, y):
        """Fit PLS model using simplified NIPALS algorithm."""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).reshape(-1, 1)

        # Center data
        self.mean_X = np.mean(X, axis=0)
        self.mean_y = np.mean(y, axis=0)
        X -= self.mean_X
        y -= self.mean_y

        n_samples, n_features = X.shape
        self.W, self.P, self.Q = [], [], []

        for _ in range(self.n_components):
            # Initial weight vector
            w = X.T @ y
            w /= np.linalg.norm(w)

            # Latent score
            t = X @ w
            p = X.T @ t / (t.T @ t)
            q = (y.T @ t) / (t.T @ t)

            # Deflation
            X -= t @ p.T
            y -= t * q

            self.W.append(w)
            self.P.append(p)
            self.Q.append(q)

        self.W = np.hstack(self.W)
        self.P = np.hstack(self.P)
        self.Q = np.hstack(self.Q)

        # Compute regression coefficients
        self.coef_ = self.W @ np.linalg.inv(self.P.T @ self.W) @ self.Q.T
        return self

    def predict(self, X):
        """Predict target using trained PLS model."""
        X = np.array(X, dtype=float) - self.mean_X
        return X @ self.coef_ + self.mean_y
