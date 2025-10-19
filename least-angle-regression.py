import numpy as np

def least_angle_regression(X, y, max_steps=None, tol=1e-6):
    """
    Performs Least Angle Regression (LARS) from scratch.

    Parameters:
        X (array-like): 2D array of predictors (m x n)
        y (array-like): 1D target array (m,)
        max_steps (int): Maximum number of steps/features to include
        tol (float): Small tolerance for stopping condition

    Returns:
        coefs (ndarray): Coefficient vector for selected features
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n_samples, n_features = X.shape
    max_steps = max_steps or n_features

    # Standardize features
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_std[X_std == 0] = 1
    X = (X - X_mean) / X_std
    y_mean = y.mean()
    y = y - y_mean

    # Initialization
    coefs = np.zeros(n_features)
    active, inactive = [], list(range(n_features))
    residual = y.copy()

    for _ in range(max_steps):
        corr = X.T @ residual
        j = np.argmax(np.abs(corr[inactive]))  # most correlated feature
        j = inactive[j]
        active.append(j)
        inactive.remove(j)

        # Solve least squares on active set
        X_active = X[:, active]
        beta_active = np.linalg.pinv(X_active) @ y
        residual = y - X_active @ beta_active

        # Stop if residuals are small
        if np.linalg.norm(residual) < tol:
            break

        coefs[active] = beta_active.flatten()

    # Undo standardization
    coefs = coefs / X_std
    intercept = y_mean - np.dot(X_mean, coefs)

    return intercept, coefs
