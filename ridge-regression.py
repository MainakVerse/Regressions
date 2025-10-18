import numpy as np

def ridge_regression(X, y, alpha=1.0):
    """
    Performs Ridge Regression (L2 Regularization) using the Normal Equation.

    Parameters:
        X (array-like): 2D array of shape (m, n) with m samples and n features.
        y (array-like): 1D array of target values (m,).
        alpha (float): Regularization strength (λ). Default = 1.0.

    Returns:
        beta (ndarray): Regression coefficients (including intercept)
        predict (function): Function to predict y for new X
    """
    X = np.array(X)
    y = np.array(y)

    # Add bias (intercept) term
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    n = X_b.shape[1]

    # Ridge Regression Normal Equation: (XᵀX + αI)⁻¹ Xᵀy
    I = np.eye(n)
    I[0, 0] = 0  # Don't regularize the intercept term
    beta = np.linalg.inv(X_b.T @ X_b + alpha * I) @ X_b.T @ y

    # Prediction function
    def predict(X_new):
        X_new = np.array(X_new)
        X_new_b = np.c_[np.ones((X_new.shape[0], 1)), X_new]
        return X_new_b @ beta

    return beta, predict
