import numpy as np

class PrincipalComponentRegressionScratch:
    def __init__(self, n_components=None):
        """
        Principal Component Regression (PCR) from scratch.

        Parameters:
            n_components (int): Number of principal components to keep
        """
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.coef_ = None

    def _standardize(self, X):
        """Mean-center data."""
        self.mean_ = np.mean(X, axis=0)
        return X - self.mean_

    def _pca(self, X):
        """Perform PCA via SVD."""
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        self.components_ = Vt[:self.n_components]
        return X @ self.components_.T

    def fit(self, X, y):
        """Fit PCR model."""
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        X_std = self._standardize(X)

        # Step 1: Apply PCA
        if self.n_components is None:
            self.n_components = X.shape[1]
        Z = self._pca(X_std)

        # Step 2: Linear Regression on PCs
        Z_b = np.c_[np.ones((len(Z), 1)), Z]
        self.coef_ = np.linalg.pinv(Z_b.T @ Z_b) @ (Z_b.T @ y)
        return self

    def predict(self, X):
        """Predict using trained PCR model."""
        X_std = X - self.mean_
        Z = X_std @ self.components_.T
        Z_b = np.c_[np.ones((len(Z), 1)), Z]
        return Z_b @ self.coef_
