import numpy as np

class LocallyWeightedRegressionScratch:
    def __init__(self, tau=1.0):
        """
        Locally Weighted Regression (LWR) from scratch.

        Parameters:
            tau (float): Bandwidth parameter controlling locality.
                         Smaller tau → more local influence.
        """
        self.tau = tau
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Store training data."""
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y, dtype=float)
        return self

    def _weights(self, x):
        """Compute Gaussian weights for each training example."""
        diffs = self.X_train - x
        return np.exp(-np.sum(diffs ** 2, axis=1) / (2 * self.tau ** 2))

    def predict_one(self, x):
        """Predict a single sample using weighted least squares."""
        X = np.c_[np.ones(len(self.X_train)), self.X_train]
        x_vec = np.array([1, *x])
        W = np.diag(self._weights(x))

        try:
            theta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ self.y_train)
            return x_vec @ theta
        except np.linalg.LinAlgError:
            # In case matrix is singular
            return np.mean(self.y_train)

    def predict(self, X):
        """Predict for multiple samples."""
        X = np.array(X, dtype=float)
        return np.array([self.predict_one(x) for x in X])
