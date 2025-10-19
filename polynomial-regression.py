import numpy as np

def polynomial_regression(X, y, degree=2):
    """
    Performs Polynomial Regression using the Normal Equation.

    Parameters:
        X (array-like): 1D or 2D feature array of shape (m,) or (m,1)
        y (array-like): Target values of shape (m,)
        degree (int): Degree of the polynomial (default=2)

    Returns:
        beta (ndarray): Polynomial coefficients (including intercept)
        predict (function): Function to predict y for new x
    """
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)

    # Expand features: [1, x, x^2, ..., x^degree]
    X_poly = np.hstack([X ** i for i in range(0, degree + 1)])

    # Compute coefficients
    beta = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y

    # Prediction function
    def predict(x_new):
        x_new = np.array(x_new).reshape(-1, 1)
        x_new_poly = np.hstack([x_new ** i for i in range(0, degree + 1)])
        return x_new_poly @ beta

    return beta, predict
