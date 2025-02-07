import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from transformers.pca_transformer import PCATransformer
from transformers.pacmap_transformer import PacMAPTransformer

def test_pacmap_transformer():
    transformer = PacMAPTransformer(n_components=2)
    X = np.random.rand(100, 10)  # Random dataset
    transformer.fit(X)
    transformed = transformer.transform(X)
    assert transformed.shape == (100, 2)  # Ensure correct output shape

def test_pca_transformer():
    transformer = PCATransformer(n_components=5)
    X = np.random.rand(100, 10)  # Random dataset
    transformer.fit(X)
    transformed = transformer.transform(X)
    assert transformed.shape == (100, 5)  # Ensure correct output shape
