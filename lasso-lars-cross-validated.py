import numpy as np

def lasso_lars_cv(X, y, alphas=None, k=5, max_iter=500, tol=1e-6):
    """
    Cross-validated LassoLars to automatically select the best alpha.

    Parameters:
        X (array-like): Feature matrix (m x n)
        y (array-like): Target vector (m,)
        alphas (list): Candidate alpha values for CV
        k (int): Number of folds for cross-validation
        max_iter (int): Max iterations for each model
        tol (float): Convergence tolerance

    Returns:
        best_alpha (float): Best alpha found via CV
        best_intercept (float): Intercept of the best model
        best_coefs (ndarray): Coefficients of the best model
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n_samples = len(y)
    if alphas is None:
        alphas = np.linspace(0.001, 1, 10)

    # Helper: split indices for k folds
    def kfold_indices(n, k):
        indices = np.arange(n)
        np.random.shuffle(indices)
        return np.array_split(indices, k)

    folds = kfold_indices(n_samples, k)
    avg_mse = []

    for alpha in alphas:
        fold_errors = []
        for i in range(k):
            test_idx = folds[i]
            train_idx = np.hstack([folds[j] for j in range(k) if j != i])

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Train LassoLars
            intercept, coefs = lasso_lars(X_train, y_train, alpha=alpha, max_iter=max_iter, tol=tol)
            y_pred = X_test @ coefs + intercept
            mse = np.mean((y_test - y_pred) ** 2)
            fold_errors.append(mse)

        avg_mse.append(np.mean(fold_errors))

    best_alpha = alphas[np.argmin(avg_mse)]
    best_intercept, best_coefs = lasso_lars(X, y, alpha=best_alpha, max_iter=max_iter, tol=tol)

    return best_alpha, best_intercept, best_coefs
