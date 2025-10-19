import numpy as np

def lasso_lars(X, y, alpha=0.1, max_iter=500, tol=1e-6):
    """
    LassoLars regression (LARS with L1 regularization) from scratch.

    Parameters:
        X (array-like): Feature matrix (m x n)
        y (array-like): Target vector (m,)
        alpha (float): Regularization strength
        max_iter (int): Maximum number of iterations
        tol (float): Convergence tolerance

    Returns:
        intercept (float): Intercept term
        coefs (ndarray): Coefficients for each feature
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n_samples, n_features = X.shape

    # Standardize X and center y
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_std[X_std == 0] = 1
    X = (X - X_mean) / X_std
    y_mean = y.mean()
    y = y - y_mean

    coefs = np.zeros(n_features)
    active, inactive = [], list(range(n_features))
    residual = y.copy()

    for _ in range(max_iter):
        corr = X.T @ residual
        j = np.argmax(np.abs(corr[inactive]))
        j = inactive[j]
        active.append(j)
        inactive.remove(j)

        X_active = X[:, active]
        beta_active = np.linalg.pinv(X_active) @ y
        residual = y - X_active @ beta_active

        # Apply L1 penalty (soft thresholding)
        for i, b in enumerate(beta_active):
            if abs(b) < alpha:
                beta_active[i] = 0
            elif b > 0:
                beta_active[i] -= alpha
            else:
                beta_active[i] += alpha

        coefs[active] = beta_active.flatten()

        if np.linalg.norm(residual) < tol:
            break

    # Undo standardization
    coefs = coefs / X_std
    intercept = y_mean - np.dot(X_mean, coefs)

    return intercept, coefs
