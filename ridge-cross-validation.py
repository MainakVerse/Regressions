import numpy as np

def ridge_regression_cv(X, y, alphas=[0.01, 0.1, 1, 10, 100], k=5):
    """
    Performs Ridge Regression with K-Fold Cross Validation.

    Parameters:
        X (array-like): 2D array of shape (m, n) — features.
        y (array-like): 1D array — target values.
        alphas (list): List of α (regularization) values to test.
        k (int): Number of folds for cross-validation.

    Returns:
        best_alpha (float): α with the lowest average validation error.
        beta (ndarray): Coefficients (including intercept) using best α.
        predict (function): Function to predict y for new X.
    """
    X = np.array(X)
    y = np.array(y)
    m = len(y)

    # Shuffle indices for cross-validation
    indices = np.arange(m)
    np.random.shuffle(indices)
    fold_size = m // k

    def ridge_beta(X_train, y_train, alpha):
        X_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
        n = X_b.shape[1]
        I = np.eye(n)
        I[0, 0] = 0
        return np.linalg.inv(X_b.T @ X_b + alpha * I) @ X_b.T @ y_train

    avg_errors = []
    for alpha in alphas:
        errors = []
        for i in range(k):
            start, end = i * fold_size, (i + 1) * fold_size
            val_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            beta = ridge_beta(X_train, y_train, alpha)
            X_val_b = np.c_[np.ones((X_val.shape[0], 1)), X_val]
            y_pred = X_val_b @ beta
            errors.append(np.mean((y_val - y_pred) ** 2))

        avg_errors.append(np.mean(errors))

    best_alpha = alphas[np.argmin(avg_errors)]
    beta = ridge_beta(X, y, best_alpha)

    # Prediction function
    def predict(X_new):
        X_new = np.array(X_new)
        X_new_b = np.c_[np.ones((X_new.shape[0], 1)), X_new]
        return X_new_b @ beta

    return best_alpha, beta, predict
