import numpy as np

def orthogonal_matching_pursuit(X, y, n_nonzero_coefs=None, tol=1e-6):
    """
    Orthogonal Matching Pursuit (OMP) implementation from scratch.

    Parameters:
        X (array-like): Feature matrix (m x n)
        y (array-like): Target vector (m,)
        n_nonzero_coefs (int): Max number of non-zero coefficients to select
        tol (float): Residual tolerance for stopping

    Returns:
        coefs (ndarray): Estimated sparse coefficients
        intercept (float): Intercept term
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n_samples, n_features = X.shape
    n_nonzero_coefs = n_nonzero_coefs or n_features

    # Center the data
    X_mean = np.mean(X, axis=0)
    y_mean = np.mean(y)
    X_centered = X - X_mean
    y_centered = y - y_mean

    residual = y_centered.copy()
    coefs = np.zeros(n_features)
    active_set = []

    for _ in range(n_nonzero_coefs):
        # Step 1: Find feature most correlated with residual
        correlations = X_centered.T @ residual
        j = np.argmax(np.abs(correlations))
        active_set.append(j)

        # Step 2: Solve least squares on active set
        X_active = X_centered[:, active_set]
        beta_active = np.linalg.pinv(X_active) @ y_centered

        # Step 3: Update residual
        residual = y_centered - X_active @ beta_active

        # Stop if residual is small enough
        if np.linalg.norm(residual) < tol:
            break

    coefs[active_set] = beta_active.flatten()
    intercept = y_mean - np.dot(X_mean, coefs)

    return intercept, coefs
