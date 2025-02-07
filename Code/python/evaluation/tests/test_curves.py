import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from evaluation.curves import (
    compute_learning_curve, compute_roc_curves, compute_precision_recall_curve
)

@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
    return X, y

@pytest.fixture
def trained_model(sample_data):
    X, y = sample_data
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

def test_compute_learning_curve(trained_model, sample_data):
    X, y = sample_data
    compute_learning_curve(trained_model, X, y, 'test_dataset', 'test_predictor', 'RandomForest')
    # Check if the function runs without errors

def test_compute_roc_curves(trained_model, sample_data):
    X, y = sample_data
    models = [trained_model]
    model_names = ['RandomForest']
    compute_roc_curves(models, model_names, X, y, ['Non-Vertebrate', 'Vertebrate'], 'test_dataset', 'test_predictor')
    # Check if the function runs without errors

def test_compute_precision_recall_curve(trained_model, sample_data):
    X, y = sample_data
    models = [trained_model]
    model_names = ['RandomForest']
    compute_precision_recall_curve(models, model_names, X, y, ['Non-Vertebrate', 'Vertebrate'], 'test_dataset', 'test_predictor')
    # Check if the function runs without errors