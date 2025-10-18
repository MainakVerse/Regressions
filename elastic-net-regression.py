import numpy as np

def elastic_net_regression(X, y, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4):
    """
    Performs Elastic Net Regression using Coordinate Descent.

    Parameters:
        X (array-like): 2D array of shape (m, n) with m samples and n features.
        y (array-like): 1D array of target values (m,).
        alpha (float): Overall regularization strength (λ).
        l1_ratio (float): Mix ratio between L1 (Lasso) and L2 (Ridge) [0, 1].
        max_iter (int): Maximum iterations for convergence.
        tol (float): Convergence tolerance.

    Returns:
        beta (ndarray): Coefficients (including intercept).
        predict (function): Function to predict y for new X.
    """
    X = np.array(X)
    y = np.array(y)
    m, n = X.shape

    # Standardize features
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_std[X_std == 0] = 1
    X_scaled = (X - X_mean) / X_std

    # Initialize parameters
    beta = np.zeros(n)
    intercept = y.mean()

    # Coordinate Descent
    for _ in range(max_iter):
        beta_old = beta.copy()

        for j in range(n):
            residual = y - (intercept + X_scaled @ beta) + beta[j] * X_scaled[:, j]
            rho = X_scaled[:, j].T @ residual

            # Soft thresholding for Elastic Net
            z = (X_scaled[:, j] ** 2).sum() + alpha * (1 - l1_ratio)
            if rho < -alpha * l1_ratio:
                beta[j] = (rho + alpha * l1_ratio) / z
            elif rho > alpha * l1_ratio:
                beta[j] = (rho - alpha * l1_ratio) / z
            else:
                beta[j] = 0.0

        # Convergence check
        if np.sum(np.abs(beta - beta_old)) < tol:
            break

    # Prediction function
    def predict(X_new):
        X_new = np.array(X_new)
        X_new_scaled = (X_new - X_mean) / X_std
        return intercept + X_new_scaled @ beta

    beta_full = np.concatenate([[intercept], beta])
    return beta_full, predict
