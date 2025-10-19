import numpy as np

def quantile_regression(X, y, tau=0.5, lr=0.01, max_iter=2000, tol=1e-6):
    """
    Quantile Regression using gradient descent.

    Parameters:
        X (array-like): Feature matrix (m, n)
        y (array-like): Target vector (m,)
        tau (float): Quantile level (0 < tau < 1), e.g. 0.5 for median
        lr (float): Learning rate
        max_iter (int): Max number of iterations
        tol (float): Tolerance for convergence

    Returns:
        intercept (float): Intercept term
        coefs (ndarray): Coefficients for features
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n_samples, n_features = X.shape

    # Add bias term
    X_b = np.c_[np.ones((n_samples, 1)), X]
    theta = np.zeros(X_b.shape[1])  # [intercept, coefficients...]

    for _ in range(max_iter):
        y_pred = X_b @ theta
        residuals = y - y_pred

        # Gradient of quantile loss
        grad = np.zeros_like(theta)
        for i in range(n_samples):
            if residuals[i] > 0:
                grad -= tau * X_b[i]
            else:
                grad += (1 - tau) * X_b[i]
        grad /= n_samples

        theta -= lr * grad

        # Stop early if gradient is small
        if np.linalg.norm(grad) < tol:
            break

    intercept = theta[0]
    coefs = theta[1:]
    return intercept, coefs
