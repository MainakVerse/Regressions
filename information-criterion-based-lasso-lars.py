import numpy as np

def lasso_lars_ic(X, y, alphas=None, criterion='aic', max_iter=500, tol=1e-6):
    """
    LassoLars model selection using Information Criterion (AIC or BIC).

    Parameters:
        X (array-like): Feature matrix (m x n)
        y (array-like): Target vector (m,)
        alphas (list): Candidate alpha values
        criterion (str): 'aic' or 'bic'
        max_iter (int): Maximum iterations for LassoLars
        tol (float): Convergence tolerance

    Returns:
        best_alpha (float): Optimal alpha minimizing AIC/BIC
        best_intercept (float): Intercept of best model
        best_coefs (ndarray): Coefficients of best model
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n_samples = len(y)
    if alphas is None:
        alphas = np.linspace(0.001, 1, 20)

    best_alpha, best_ic = None, np.inf
    best_intercept, best_coefs = 0, None

    for alpha in alphas:
        intercept, coefs = lasso_lars(X, y, alpha=alpha, max_iter=max_iter, tol=tol)
        y_pred = X @ coefs + intercept
        residual = y - y_pred
        rss = np.sum(residual ** 2)
        k = np.sum(coefs != 0)

        if criterion == 'aic':
            ic = n_samples * np.log(rss / n_samples) + 2 * k
        elif criterion == 'bic':
            ic = n_samples * np.log(rss / n_samples) + k * np.log(n_samples)
        else:
            raise ValueError("criterion must be 'aic' or 'bic'")

        if ic < best_ic:
            best_ic = ic
            best_alpha = alpha
            best_intercept = intercept
            best_coefs = coefs

    return best_alpha, best_intercept, best_coefs
