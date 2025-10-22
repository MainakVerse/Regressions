import numpy as np

class IsotonicRegressionScratch:
    def __init__(self, increasing=True):
        """
        Isotonic Regression from scratch (using PAVA).
        
        Parameters:
            increasing (bool): Whether the fitted function should be non-decreasing (default=True)
        """
        self.increasing = increasing
        self.y_ = None
        self.x_ = None

    def fit(self, X, y):
        """Fit isotonic regression model using PAVA."""
        X, y = np.array(X, dtype=float).flatten(), np.array(y, dtype=float).flatten()

        # Sort data by X to ensure monotonicity
        sort_idx = np.argsort(X)
        X, y = X[sort_idx], y[sort_idx]

        n = len(y)
        y_fit = y.copy()
        weights = np.ones(n)

        # Pool Adjacent Violators Algorithm
        i = 0
        while i < n - 1:
            if (self.increasing and y_fit[i] > y_fit[i + 1]) or (not self.increasing and y_fit[i] < y_fit[i + 1]):
                # Merge blocks
                total_weight = weights[i] + weights[i + 1]
                avg = (y_fit[i] * weights[i] + y_fit[i + 1] * weights[i + 1]) / total_weight
                y_fit[i] = avg
                weights[i] = total_weight

                # Remove merged element
                y_fit = np.delete(y_fit, i + 1)
                weights = np.delete(weights, i + 1)
                n -= 1

                # Step back if possible
                if i > 0:
                    i -= 1
            else:
                i += 1

        # Expand piecewise constant predictions
        self.x_ = np.unique(X)
        self.y_ = np.repeat(y_fit, np.bincount(np.searchsorted(self.x_, X, side='right'))[:-1])
        return self

    def predict(self, X):
        """Predict by piecewise constant interpolation."""
        X = np.array(X, dtype=float).flatten()
        preds = np.interp(X, self.x_, self.y_)
        return preds
