import numpy as np

def robust_linear_model(X, y, delta=1.0, max_iter=100, tol=1e-6):
    """
    Robust Linear Model using Iteratively Reweighted Least Squares (IRLS).

    Parameters:
        X (array-like): Feature matrix (m, n)
        y (array-like): Target vector (m,)
        delta (float): Threshold for Huber weight function
        max_iter (int): Maximum number of IRLS iterations
        tol (float): Convergence tolerance

    Returns:
        intercept (float): Intercept term
        coefs (ndarray): Coefficients for features
        weights (ndarray): Final observation weights
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n_samples, n_features = X.shape

    # Add bias column
    X_b = np.c_[np.ones((n_samples, 1)), X]
    theta = np.zeros(X_b.shape[1])  # [intercept, coefficients...]

    weights = np.ones(n_samples)

    for _ in range(max_iter):
        # Weighted least squares step
        W = np.diag(weights)
        try:
            theta_new = np.linalg.inv(X_b.T @ W @ X_b) @ X_b.T @ W @ y
        except np.linalg.LinAlgError:
            break

        residuals = y - X_b @ theta_new
        abs_res = np.abs(residuals)
        sigma = np.median(abs_res) / 0.6745  # robust std estimate

        # Huber weighting
        weights = np.where(abs_res <= delta * sigma, 1, (delta * sigma) / abs_res)

        # Check for convergence
        if np.linalg.norm(theta_new - theta) < tol:
            theta = theta_new
            break

        theta = theta_new

    intercept = theta[0]
    coefs = theta[1:]
    return intercept, coefs, weights
