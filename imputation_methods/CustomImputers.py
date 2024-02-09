import numpy as np
from sklearn.neighbors import NearestNeighbors

class CustomKNNImputer:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X):
        self.nn_model_ = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        self.nn_model_.fit(X)

    def impute(self, X):
        missing_mask = np.isnan(X)
        if not np.any(missing_mask):
            return X

        # Find indices of missing values
        missing_indices = np.where(missing_mask)

        # Replace missing values with 0 for the purpose of finding nearest neighbors
        X_zeroed = np.nan_to_num(X, nan=0)

        # Find nearest neighbors for each missing value
        distances, indices = self.nn_model_.kneighbors(X_zeroed)

        # Impute missing values with mean of nearest neighbors
        imputed_values = np.mean(np.take_along_axis(X, indices[:, 1:], axis=0), axis=1)

        # Replace missing values in original array
        X[missing_indices] = imputed_values

        return X
