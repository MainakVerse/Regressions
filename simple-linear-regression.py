import numpy as np

def simple_linear_regression(X, y):
    """
    Performs Simple Linear Regression using the least squares method.

    Parameters:
        X (array-like): Feature values (1D)
        y (array-like): Target values (1D)

    Returns:
        b0 (float): Intercept
        b1 (float): Slope
        predict (function): Function to predict y for given x
    """
    X = np.array(X)
    y = np.array(y)

    # Calculate means
    x_mean = np.mean(X)
    y_mean = np.mean(y)

    # Calculate coefficients
    numerator = np.sum((X - x_mean) * (y - y_mean))
    denominator = np.sum((X - x_mean)**2)
    b1 = numerator / denominator
    b0 = y_mean - b1 * x_mean

    # Prediction function
    def predict(x):
        return b0 + b1 * np.array(x)

    return b0, b1, predict
