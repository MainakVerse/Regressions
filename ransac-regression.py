import numpy as np

def ransac_regression(X, y, n_iter=100, sample_size=2, threshold=2.0, min_inliers=0.5):
    """
    Robust RANSAC Linear Regression implementation.

    Parameters:
        X (array-like): Feature array (m,) or (m,1)
        y (array-like): Target array (m,)
        n_iter (int): Number of random iterations
        sample_size (int): Number of points per sample
        threshold (float): Max residual error to consider an inlier
        min_inliers (float): Minimum fraction of inliers (0–1) to refit final model

    Returns:
        best_intercept (float)
        best_slope (float)
        inlier_mask (ndarray): Boolean mask for inliers
    """
    X = np.array(X, dtype=float).flatten()
    y = np.array(y, dtype=float).flatten()
    n_samples = len(X)
    best_inliers = []
    best_model = (0, 0)

    for _ in range(n_iter):
        # Random subset
        idx = np.random.choice(n_samples, sample_size, replace=False)
        X_sample, y_sample = X[idx], y[idx]

        # Fit simple linear regression on subset
        x_mean, y_mean = np.mean(X_sample), np.mean(y_sample)
        slope = np.sum((X_sample - x_mean) * (y_sample - y_mean)) / np.sum((X_sample - x_mean) ** 2)
        intercept = y_mean - slope * x_mean

        # Compute residuals and find inliers
        y_pred = intercept + slope * X
        residuals = np.abs(y - y_pred)
        inliers = residuals < threshold

        # Keep best model
        if np.sum(inliers) > np.sum(best_inliers):
            best_inliers = inliers
            best_model = (intercept, slope)

    # If enough inliers, refit final model
    if np.sum(best_inliers) >= min_inliers * n_samples:
        X_in, y_in = X[best_inliers], y[best_inliers]
        x_mean, y_mean = np.mean(X_in), np.mean(y_in)
        slope = np.sum((X_in - x_mean) * (y_in - y_mean)) / np.sum((X_in - x_mean) ** 2)
        intercept = y_mean - slope * x_mean
        best_model = (intercept, slope)

    return best_model[0], best_model[1], best_inliers
