import numpy as np

def elastic_net_cv(X, y, alphas=[0.01, 0.1, 1.0], l1_ratios=[0.2, 0.5, 0.8],
                   k=5, max_iter=1000, tol=1e-4):
    """
    Performs Elastic Net Regression with K-Fold Cross Validation.

    Parameters:
        X (array-like): 2D array (m, n)
        y (array-like): 1D array (m,)
        alphas (list): List of λ values to test
        l1_ratios (list): List of L1 ratios to test
        k (int): Number of folds
        max_iter (int): Maximum iterations
        tol (float): Convergence tolerance

    Returns:
        best_alpha (float): Best λ
        best_l1_ratio (float): Best L1 ratio
        beta (ndarray): Coefficients (including intercept)
        predict (function): Prediction function
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

    def fit_elastic(Xt, yt, alpha, l1_ratio):
        beta = np.zeros(n)
        intercept = yt.mean()
        for _ in range(max_iter):
            beta_old = beta.copy()
            for j in range(n):
                residual = yt - (intercept + Xt @ beta) + beta[j] * Xt[:, j]
                rho = Xt[:, j].T @ residual
                z = (Xt[:, j] ** 2).sum() + alpha * (1 - l1_ratio)
                if rho < -alpha * l1_ratio:
                    beta[j] = (rho + alpha * l1_ratio) / z
                elif rho > alpha * l1_ratio:
                    beta[j] = (rho - alpha * l1_ratio) / z
                else:
                    beta[j] = 0.0
            if np.sum(np.abs(beta - beta_old)) < tol:
                break
        return intercept, beta

    # Cross-validation grid search
    best_error = np.inf
    best_alpha = None
    best_l1_ratio = None

    for alpha in alphas:
        for l1_ratio in l1_ratios:
            fold_errors = []
            for i in range(k):
                start, end = i * fold_size, (i + 1) * fold_size
                val_idx = indices[start:end]
                train_idx = np.concatenate([indices[:start], indices[end:]])

                X_train, y_train = X_scaled[train_idx], y[train_idx]
                X_val, y_val = X_scaled[val_idx], y[val_idx]

                intercept, beta = fit_elastic(X_train, y_train, alpha, l1_ratio)
                y_pred = intercept + X_val @ beta
                fold_errors.append(np.mean((y_val - y_pred) ** 2))
            avg_error = np.mean(fold_errors)

            if avg_error < best_error:
                best_error = avg_error
                best_alpha, best_l1_ratio = alpha, l1_ratio

    # Final model with best params
    intercept, beta = fit_elastic(X_scaled, y, best_alpha, best_l1_ratio)

    def predict(X_new):
        X_new = np.array(X_new)
        X_new_scaled = (X_new - X_mean) / X_std
        return intercept + X_new_scaled @ beta

    beta_full = np.concatenate([[intercept], beta])
    return best_alpha, best_l1_ratio, beta_full, predict
