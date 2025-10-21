import numpy as np

class KNNRegressorScratch:
    def __init__(self, n_neighbors=5, metric='euclidean'):
        """
        K-Nearest Neighbors (KNN) Regressor from scratch.
        
        Parameters:
            n_neighbors (int): Number of neighbors to average.
            metric (str): Distance metric ('euclidean' or 'manhattan').
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Store the training data (no real training phase)."""
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y, dtype=float)
        return self

    def _distance(self, x1, x2):
        """Compute distance between two samples."""
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError("Unsupported metric")

    def predict(self, X):
        """Predict by averaging target values of nearest neighbors."""
        X = np.array(X, dtype=float)
        predictions = []

        for x in X:
            # Compute all distances to training points
            distances = np.array([self._distance(x, x_train) for x_train in self.X_train])
            # Get indices of k nearest neighbors
            neighbor_idx = np.argsort(distances)[:self.n_neighbors]
            # Average their y values
            y_pred = np.mean(self.y_train[neighbor_idx])
            predictions.append(y_pred)

        return np.array(predictions)
