import numpy as np

class RadiusNeighborsRegressorScratch:
    def __init__(self, radius=1.0, metric='euclidean'):
        """
        Radius Neighbors Regressor from scratch.
        
        Parameters:
            radius (float): Distance threshold for neighbor inclusion.
            metric (str): Distance metric ('euclidean' or 'manhattan').
        """
        self.radius = radius
        self.metric = metric
        self.X_train = None
        self.y_train = None
        self.y_mean = None

    def fit(self, X, y):
        """Store the training data."""
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y, dtype=float)
        self.y_mean = np.mean(y)
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
        """Predict by averaging targets of points within radius."""
        X = np.array(X, dtype=float)
        predictions = []

        for x in X:
            distances = np.array([self._distance(x, x_train) for x_train in self.X_train])
            within_radius = distances <= self.radius

            if np.any(within_radius):
                y_pred = np.mean(self.y_train[within_radius])
            else:
                # Fallback to global mean if no neighbors
                y_pred = self.y_mean

            predictions.append(y_pred)

        return np.array(predictions)
