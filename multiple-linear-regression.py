import numpy as np

def multiple_linear_regression(X, y):
    """
    Performs Multiple Linear Regression using the Normal Equation.

    Parameters:
        X (array-like): 2D array of shape (m, n) with m samples and n features.
        y (array-like): 1D array of target values (m,).

    Returns:
        beta (ndarray): Regression coefficients (including intercept)
        predict (function): Function to predict y for new X
    """
    X = np.array(X)
    y = np.array(y)

    # Add bias (intercept) term
    X_b = np.c_[np.ones((X.shape[0], 1)), X]

    # Calculate coefficients using the Normal Equation
    beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    # Prediction function
    def predict(X_new):
        X_new = np.array(X_new)
        X_new_b = np.c_[np.ones((X_new.shape[0], 1)), X_new]
        return X_new_b @ beta

    return beta, predict
