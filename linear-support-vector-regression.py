import numpy as np

def linear_svr(X, y, C=1.0, epsilon=0.1, lr=0.001, max_iter=2000, tol=1e-6):
    """
    Linear Support Vector Regression (LinearSVR) using gradient descent.

    Parameters:
        X (array-like): Feature matrix (m, n)
        y (array-like): Target vector (m,)
        C (float): Regularization strength
        epsilon (float): Epsilon-insensitive margin
        lr (float): Learning rate
        max_iter (int): Max training iterations
        tol (float): Convergence tolerance

    Returns:
        intercept (float): Bias term
        weights (ndarray): Coefficient vector
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n_samples, n_features = X.shape

    w = np.zeros(n_features)
    b = 0.0

    for _ in range(max_iter):
        y_pred = X @ w + b
        residuals = y - y_pred

        # Gradient initialization
        grad_w = w.copy()  # L2 regularization gradient
        grad_b = 0.0

        for i in range(n_samples):
            r = residuals[i]
            if abs(r) > epsilon:
                grad_w -= C * np.sign(r) * X[i]
                grad_b -= C * np.sign(r)

        # Update parameters
        w -= lr * grad_w
        b -= lr * grad_b

        # Convergence check
        if np.linalg.norm(grad_w) < tol:
            break

    return b, w
