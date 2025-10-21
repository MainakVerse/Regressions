import numpy as np

class GeneralizedLinearModelScratch:
    def __init__(self, family="gaussian", link="identity", max_iter=100, tol=1e-6):
        """
        Generalized Linear Model (GLM) from scratch.

        Parameters:
            family (str): 'gaussian', 'poisson', or 'binomial'
            link (str): 'identity', 'log', or 'logit'
            max_iter (int): Maximum IRLS iterations
            tol (float): Convergence tolerance
        """
        self.family = family
        self.link = link
        self.max_iter = max_iter
        self.tol = tol
        self.beta = None

    def _link(self, mu):
        """Link function g(mu)."""
        if self.link == "identity":
            return mu
        elif self.link == "log":
            return np.log(mu)
        elif self.link == "logit":
            return np.log(mu / (1 - mu))
        else:
            raise ValueError("Unsupported link")

    def _inv_link(self, eta):
        """Inverse link function g⁻¹(eta)."""
        if self.link == "identity":
            return eta
        elif self.link == "log":
            return np.exp(eta)
        elif self.link == "logit":
            return 1 / (1 + np.exp(-eta))
        else:
            raise ValueError("Unsupported inverse link")

    def fit(self, X, y):
        """Fit GLM using IRLS."""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n_samples, n_features = X.shape

        X_b = np.c_[np.ones((n_samples, 1)), X]
        self.beta = np.zeros(X_b.shape[1])

        for _ in range(self.max_iter):
            eta = X_b @ self.beta
            mu = self._inv_link(eta)

            # Variance function per family
            if self.family == "gaussian":
                var_mu = np.ones_like(mu)
            elif self.family == "poisson":
                var_mu = mu
            elif self.family == "binomial":
                var_mu = mu * (1 - mu)
            else:
                raise ValueError("Unsupported family")

            # Derivative of link function
            if self.link == "identity":
                d_eta = np.ones_like(mu)
            elif self.link == "log":
                d_eta = 1 / mu
            elif self.link == "logit":
                d_eta = 1 / (mu * (1 - mu))

            # Working response
            z = eta + (y - mu) * d_eta

            # Weights
            W = np.diag((d_eta ** 2) / var_mu)

            # Update coefficients (Weighted Least Squares)
            beta_new = np.linalg.pinv(X_b.T @ W @ X_b) @ (X_b.T @ W @ z)

            if np.linalg.norm(beta_new - self.beta) < self.tol:
                break

            self.beta = beta_new

        return self

    def predict(self, X):
        """Predict mean response."""
        X = np.array(X, dtype=float)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        eta = X_b @ self.beta
        return self._inv_link(eta)
