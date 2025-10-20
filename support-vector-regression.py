import numpy as np

def support_vector_regression(X, y, C=1.0, epsilon=0.1, lr=0.001, max_iter=1000, tol=1e-5):
    """
    Linear Support Vector Regression (SVR) using gradient descent.

    Parameters:
        X (array-like): Feature matrix (m, n)
        y (array-like): Target vector (m,)
        C (float): Regularization parameter
        epsilon (float): Epsilon-insensitive margin
        lr (float): Learning rate
        max_iter (int): Maximum iterations
        tol (float): Convergence tolerance

    Returns:
        intercept (float)
        weights (ndarray)
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n_samples, n_features = X.shape

    w = np.zeros(n_features)
    b = 0.0

    for _ in range(max_iter):
        y_pred = X @ w + b
        error = y_pred - y

        grad_w = np.zeros_like(w)
        grad_b = 0

        for i in range(n_samples):
            if abs(error[i]) > epsilon:
                grad_w += C * np.sign(error[i]) * X[i]
                grad_b += C * np.sign(error[i])

        grad_w += w  # regularization term

        # Update weights and bias
        w -= lr * grad_w
        b -= lr * grad_b

        # Stop if gradients small
        if np.linalg.norm(grad_w) < tol:
            break

    return b, w
