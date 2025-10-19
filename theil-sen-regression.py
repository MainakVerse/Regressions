import numpy as np

def theil_sen_regression(X, y):
    """
    Theil–Sen robust linear regression from scratch.

    Parameters:
        X (array-like): Feature array (m,) or (m, 1)
        y (array-like): Target array (m,)

    Returns:
        intercept (float): Intercept term
        slope (float): Slope coefficient
    """
    X = np.array(X, dtype=float).flatten()
    y = np.array(y, dtype=float).flatten()
    n = len(X)

    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if X[j] != X[i]:
                slope = (y[j] - y[i]) / (X[j] - X[i])
                slopes.append(slope)

    if len(slopes) == 0:
        raise ValueError("All x values are identical — cannot compute slope")

    b1 = np.median(slopes)
    b0 = np.median(y - b1 * X)

    return b0, b1
