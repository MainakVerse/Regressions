import numpy as np

def lasso_regression_cv(X, y, alphas=[0.001, 0.01, 0.1, 1.0], k=5, max_iter=1000, tol=1e-4):
    """
    Performs Lasso Regression with K-Fold Cross Validation using Coordinate Descent.

    Parameters:
        X (array-like): 2D array of shape (m, n) with m samples and n features.
        y (array-like): 1D array of target values (m,).
        alphas (list): List of λ (regularization) values to test.
        k (int): Number of folds for cross-validation.
        max_iter (int): Maximum number of iterations for convergence.
        tol (float): Convergence tolerance.

    Returns:
        best_alpha (float): Regularization value with lowest validation error.
        beta (ndarray): Coefficients (including intercept) for best α.
        predict (function): Function to predict y for new X.
    """
    X = np.array(X)
    y = np.array(y)
    m, n = X.shape

    # Standardize features
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_std[X_std == 0] = 1
    X_scaled = (X - X_mean) / X_std

    # Shuffle data
    indices = np.arange(m)
    np.random.shuffle(indices)
    fold_size = m // k

    def soft_threshold(rho, alpha):
        if rho < -alpha:
            return rho + alpha
        elif rho > alpha:
            return rho - alpha
        else:
            return 0.0

    def fit_lasso(X_train, y_train, alpha):
        beta = np.zeros(n)
        intercept = y_train.mean()

        for _ in range(max_iter):
            beta_old = beta.copy()
            for j in range(n):
                residual = y_train - (intercept + X_train @ beta) + beta[j] * X_train[:, j]
                rho = X_train[:, j].T @ residual
                beta[j] = soft_threshold(rho, alpha / 2) / (X_train[:, j] ** 2).sum()
            if np.sum(np.abs(beta - beta_old)) < tol:
                break
        return intercept, beta

    # Cross-validation
    avg_errors = []
    for alpha in alphas:
        errors = []
        for i in range(k):
            start, end = i * fold_size, (i + 1) * fold_size
            val_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])

            X_train, y_train = X_scaled[train_idx], y[train_idx]
            X_val, y_val = X_scaled[val_idx], y[val_idx]

            intercept, beta = fit_lasso(X_train, y_train, alpha)
            y_pred = intercept + X_val @ beta
            errors.append(np.mean((y_val - y_pred) ** 2))
        avg_errors.append(np.mean(errors))

    best_alpha = alphas[np.argmin(avg_errors)]
    intercept, beta = fit_lasso(X_scaled, y, best_alpha)

    # Prediction function
    def predict(X_new):
        X_new = np.array(X_new)
        X_new_scaled = (X_new - X_mean) / X_std
        return intercept + X_new_scaled @ beta

    beta_full = np.concatenate([[intercept], beta])
    return best_alpha, beta_full, predict
