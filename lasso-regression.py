import numpy as np

def lasso_regression(X, y, alpha=0.1, max_iter=1000, tol=1e-4):
    """
    Performs Lasso Regression (L1 Regularization) using Coordinate Descent.

    Parameters:
        X (array-like): 2D array of shape (m, n) with m samples and n features.
        y (array-like): 1D array of target values (m,).
        alpha (float): Regularization strength (λ). Default = 0.1.
        max_iter (int): Maximum number of iterations. Default = 1000.
        tol (float): Convergence tolerance. Default = 1e-4.

    Returns:
        beta (ndarray): Regression coefficients (including intercept)
        predict (function): Function to predict y for new X
    """
    X = np.array(X)
    y = np.array(y)
    m, n = X.shape

    # Standardize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1
    X_scaled = (X - X_mean) / X_std

    # Initialize coefficients
    beta = np.zeros(n)
    intercept = y.mean()

    for _ in range(max_iter):
        beta_old = beta.copy()

        for j in range(n):
            # Partial residual
            residual = y - (intercept + X_scaled @ beta) + beta[j] * X_scaled[:, j]
            rho = X_scaled[:, j].T @ residual

            # Soft thresholding
            if rho < -alpha / 2:
                beta[j] = (rho + alpha / 2) / (X_scaled[:, j] ** 2).sum()
            elif rho > alpha / 2:
                beta[j] = (rho - alpha / 2) / (X_scaled[:, j] ** 2).sum()
            else:
                beta[j] = 0.0

        # Check for convergence
        if np.sum(np.abs(beta - beta_old)) < tol:
            break

    # Prediction function
    def predict(X_new):
        X_new = np.array(X_new)
        X_new_scaled = (X_new - X_mean) / X_std
        return intercept + X_new_scaled @ beta

    # Combine intercept and coefficients
    beta_full = np.concatenate([[intercept], beta])

    return beta_full, predict
