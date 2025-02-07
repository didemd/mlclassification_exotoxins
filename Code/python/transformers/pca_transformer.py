"""
Standard Principal Component Analysis (PCA) for dimensionality reduction 
wraping Scikit-learn’s built-in PCA into a custom transformer.
"""
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA

class PCATransformer(TransformerMixin, BaseEstimator):
    def __init__(self, n_components, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)

    def fit(self, X, y=None):
        self.pca.fit(X)
        return self

    def transform(self, X):
        return self.pca.transform(X)
