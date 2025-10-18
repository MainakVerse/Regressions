import numpy as np

def bayesian_ridge_regression(X, y, alpha_init=1.0, lambda_init=1.0, max_iter=500, tol=1e-4):
    """
    Performs Bayesian Ridge Regression using iterative evidence maximization.

    Parameters:
        X (array-like): 2D array of shape (m, n)
        y (array-like): 1D array of target values (m,)
        alpha_init (float): Initial precision of noise (1/variance)
        lambda_init (float): Initial precision of weights (prior strength)
        max_iter (int): Maximum number of iterations
        tol (float): Convergence tolerance

    Returns:
        beta_mean (ndarray): Posterior mean of coefficients (including intercept)
        alpha (float): Final noise precision
        lambda_ (float): Final weight precision
        predict (function): Prediction function for new data
    """
    X = np.array(X)
    y = np.array(y)
    m, n = X.shape

    # Add bias column
    X_b = np.c_[np.ones((m, 1)), X]

    # Initialize parameters
    alpha = alpha_init
    lambda_ = lambda_init
    beta_mean = np.zeros(n + 1)

    for _ in range(max_iter):
        beta_old = beta_mean.copy()

        # Posterior covariance and mean
        A = lambda_ * np.eye(n + 1) + alpha * (X_b.T @ X_b)
        A_inv = np.linalg.inv(A)
        beta_mean = alpha * A_inv @ X_b.T @ y

        # Compute effective number of parameters (γ)
        gamma = np.sum(alpha * np.linalg.eigvalsh(X_b.T @ X_b @ A_inv))

        # Update hyperparameters
        residual = y - X_b @ beta_mean
        alpha = (m - gamma) / (residual @ residual)
        lambda_ = gamma / (beta_mean @ beta_mean)

        # Check convergence
        if np.linalg.norm(beta_mean - beta_old) < tol:
            break

    # Prediction function
    def predict(X_new):
        X_new = np.array(X_new)
        X_new_b = np.c_[np.ones((X_new.shape[0], 1)), X_new]
        return X_new_b @ beta_mean

    return beta_mean, alpha, lambda_, predict
