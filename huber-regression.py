import numpy as np

def huber_regression(X, y, delta=1.0, lr=0.01, max_iter=1000, tol=1e-6):
    """
    Robust Huber Regression using gradient descent.

    Parameters:
        X (array-like): Feature matrix (m x n)
        y (array-like): Target vector (m,)
        delta (float): Threshold parameter for Huber loss
        lr (float): Learning rate for gradient descent
        max_iter (int): Max number of iterations
        tol (float): Convergence tolerance

    Returns:
        intercept (float): Intercept term
        coefs (ndarray): Regression coefficients
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n_samples, n_features = X.shape

    # Add bias term
    X_b = np.c_[np.ones((n_samples, 1)), X]
    theta = np.zeros(X_b.shape[1])  # [b0, b1, b2, ...]

    for _ in range(max_iter):
        y_pred = X_b @ theta
        residuals = y - y_pred

        # Compute derivative of Huber loss
        gradient = np.zeros_like(theta)
        for i in range(n_samples):
            r = residuals[i]
            if abs(r) <= delta:
                grad = -r * X_b[i]
            else:
                grad = -delta * np.sign(r) * X_b[i]
            gradient += grad
        gradient /= n_samples

        theta -= lr * gradient

        # Stop if gradient is small
        if np.linalg.norm(gradient) < tol:
            break

    intercept = theta[0]
    coefs = theta[1:]
    return intercept, coefs
