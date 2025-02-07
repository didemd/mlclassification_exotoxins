"""A dimensionality reduction method optimized for visualizing high-dimensional data."""
import pacmap
from sklearn.base import TransformerMixin, BaseEstimator

class PacMAPTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=42):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.MN_ratio = MN_ratio
        self.FP_ratio = FP_ratio
        self.random_state = random_state

    def fit(self, X, y=None):
        self._pacmap = pacmap.PaCMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            MN_ratio=self.MN_ratio,
            FP_ratio=self.FP_ratio,
            random_state=self.random_state
        )
        self._pacmap.fit(X)
        self._fitted_data = X  # Cache the fitted dataset for transformation
        return self

    def transform(self, X):
        if not hasattr(self, '_pacmap'):
            raise RuntimeError("PacMAPTransformer has not been fitted yet.")
        return self._pacmap.transform(X, basis=self._fitted_data)

    def get_params(self, deep=True):
        return {
            'n_components': self.n_components,
            'n_neighbors': self.n_neighbors,
            'MN_ratio': self.MN_ratio,
            'FP_ratio': self.FP_ratio,
            'random_state': self.random_state
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self