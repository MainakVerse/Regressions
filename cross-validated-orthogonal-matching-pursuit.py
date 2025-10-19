import numpy as np

def orthogonal_matching_pursuit_cv(X, y, max_nonzero=10, k=5, tol=1e-6):
    """
    Cross-validated Orthogonal Matching Pursuit (OMP-CV) from scratch.

    Parameters:
        X (array-like): Feature matrix (m x n)
        y (array-like): Target vector (m,)
        max_nonzero (int): Max number of non-zero coefficients to test
        k (int): Number of CV folds
        tol (float): Tolerance for convergence

    Returns:
        best_k (int): Optimal number of non-zero coefficients
        best_intercept (float): Intercept of best model
        best_coefs (ndarray): Coefficients of best model
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n_samples = len(y)

    # Helper to split folds
    def kfold_indices(n, k):
        indices = np.arange(n)
        np.random.shuffle(indices)
        return np.array_split(indices, k)

    folds = kfold_indices(n_samples, k)
    avg_mse = []

    for n_nonzero in range(1, max_nonzero + 1):
        fold_errors = []

        for i in range(k):
            test_idx = folds[i]
            train_idx = np.hstack([folds[j] for j in range(k) if j != i])

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            intercept, coefs = orthogonal_matching_pursuit(
                X_train, y_train, n_nonzero_coefs=n_nonzero, tol=tol
            )

            y_pred = X_test @ coefs + intercept
            mse = np.mean((y_test - y_pred) ** 2)
            fold_errors.append(mse)

        avg_mse.append(np.mean(fold_errors))

    # Select best number of nonzero coefficients
    best_k = np.argmin(avg_mse) + 1
    best_intercept, best_coefs = orthogonal_matching_pursuit(
        X, y, n_nonzero_coefs=best_k, tol=tol
    )

    return best_k, best_intercept, best_coefs
