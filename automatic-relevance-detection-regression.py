import numpy as np

def ard_regression(X, y, alpha_init=1.0, lambda_init=1.0, max_iter=500, tol=1e-4):
    """
    Performs ARD (Automatic Relevance Determination) Regression using
    iterative evidence maximization.

    Parameters:
        X (array-like): 2D array of shape (m, n)
        y (array-like): 1D array of target values (m,)
        alpha_init (float): Initial noise precision (1/variance)
        lambda_init (float): Initial prior precision for each weight
        max_iter (int): Maximum iterations
        tol (float): Convergence tolerance

    Returns:
        beta_mean (ndarray): Posterior mean of coefficients (including intercept)
        alpha (float): Final noise precision
        lambdas (ndarray): Final individual weight precisions
        predict (function): Function to predict y for new X
    """
    X = np.array(X)
    y = np.array(y)
    m, n = X.shape

    # Add intercept term
    X_b = np.c_[np.ones((m, 1)), X]

    # Initialize parameters
    alpha = alpha_init
    lambdas = np.full(n + 1, lambda_init)
    lambdas[0] = 0  # No regularization on intercept
    beta_mean = np.zeros(n + 1)

    for _ in range(max_iter):
        beta_old = beta_mean.copy()

        # Compute posterior covariance and mean
        A = np.diag(lambdas) + alpha * (X_b.T @ X_b)
        A_inv = np.linalg.inv(A)
        beta_mean = alpha * A_inv @ X_b.T @ y

        # Compute posterior covariance diagonal
        Sigma_diag = np.diag(A_inv)

        # Update individual precisions (λⱼ)
        gamma = 1 - lambdas * Sigma_diag
        lambdas = gamma / (beta_mean**2 + 1e-8)
        lambdas[0] = 0  # exclude intercept

        # Update noise precision α
        residual = y - X_b @ beta_mean
        alpha = (m - np.sum(gamma)) / (residual @ residual + 1e-8)

        # Convergence check
        if np.linalg.norm(beta_mean - beta_old) < tol:
            break

    # Prediction function
    def predict(X_new):
        X_new = np.array(X_new)
        X_new_b = np.c_[np.ones((X_new.shape[0], 1)), X_new]
        return X_new_b @ beta_mean

    return beta_mean, alpha, lambdas, predict
