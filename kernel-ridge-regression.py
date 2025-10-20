import numpy as np

def kernel_ridge_regression(X, y, lambda_=1.0, kernel="rbf", gamma=0.1, degree=3):
    """
    Kernel Ridge Regression (KRR) from scratch.

    Parameters:
        X (array-like): Training features (m x n)
        y (array-like): Target values (m,)
        lambda_ (float): Regularization parameter
        kernel (str): 'linear', 'poly', or 'rbf'
        gamma (float): Kernel coefficient for RBF/poly
        degree (int): Degree for polynomial kernel

    Returns:
        predict (function): Function for making predictions on new data
        alpha (ndarray): Dual coefficients
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    # Define kernel function
    def compute_kernel(A, B):
        if kernel == "linear":
            return A @ B.T
        elif kernel == "poly":
            return (A @ B.T + 1) ** degree
        elif kernel == "rbf":
            sq_dists = np.sum(A**2, axis=1, keepdims=True) + np.sum(B**2, axis=1) - 2 * A @ B.T
            return np.exp(-gamma * sq_dists)
        else:
            raise ValueError("Unsupported kernel type.")

    # Compute kernel matrix K
    K = compute_kernel(X, X)

    # Solve for alpha: (K + λI)α = y
    alpha = np.linalg.inv(K + lambda_ * np.eye(len(X))) @ y

    # Prediction function
    def predict(X_new):
        K_new = compute_kernel(X_new, X)
        return K_new @ alpha

    return predict, alpha
