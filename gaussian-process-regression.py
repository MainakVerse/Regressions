import numpy as np

class GaussianProcessRegressorScratch:
    def __init__(self, kernel="rbf", sigma_n=1e-2, gamma=1.0, length_scale=1.0):
        """
        Gaussian Process Regressor (from scratch).

        Parameters:
            kernel (str): 'rbf' or 'linear'
            sigma_n (float): Noise variance term
            gamma (float): RBF kernel parameter (controls width)
            length_scale (float): Scale parameter for RBF
        """
        self.kernel = kernel
        self.sigma_n = sigma_n
        self.gamma = gamma
        self.length_scale = length_scale
        self.X_train = None
        self.y_train = None
        self.K_inv = None

    def _kernel_func(self, X1, X2):
        """Compute kernel matrix."""
        if self.kernel == "linear":
            return X1 @ X2.T
        elif self.kernel == "rbf":
            sq_dists = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * (X1 @ X2.T)
            return np.exp(-sq_dists / (2 * self.length_scale**2))
        else:
            raise ValueError("Unsupported kernel type")

    def fit(self, X, y):
        """Fit Gaussian Process to training data."""
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y, dtype=float)

        K = self._kernel_func(self.X_train, self.X_train)
        self.K_inv = np.linalg.inv(K + self.sigma_n**2 * np.eye(len(self.X_train)))
        return self

    def predict(self, X_test, return_std=False):
        """Compute GP posterior mean and (optional) standard deviation."""
        X_test = np.array(X_test, dtype=float)
        K_s = self._kernel_func(self.X_train, X_test)
        K_ss = self._kernel_func(X_test, X_test)

        mu_s = K_s.T @ self.K_inv @ self.y_train
        cov_s = K_ss - K_s.T @ self.K_inv @ K_s

        if return_std:
            std_s = np.sqrt(np.diag(cov_s))
            return mu_s, std_s
        return mu_s
