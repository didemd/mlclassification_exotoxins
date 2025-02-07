import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from models.training import (
    validate_inputs,
    create_pipeline,
    perform_grid_search,
    train_random_forest,
    train_logistic_regression,
    train_svm,
    train_k_neighbors,
    PARAM_GRIDS
)

@pytest.fixture
def sample_data():
    """
    Generate sample training data for tests.
    """
    X_train = np.random.rand(100, 20)  # 100 samples, 20 features
    y_train = np.random.choice([0, 1], size=100)  # Binary labels
    return X_train, y_train


def test_validate_inputs_valid(sample_data):
    """
    Test validate_inputs with valid input data.
    """
    X_train, y_train = sample_data
    validate_inputs(X_train, y_train)  # Should not raise any exception


def test_validate_inputs_invalid():
    """
    Test validate_inputs with invalid input data.
    """
    with pytest.raises(ValueError, match="Training data cannot be None."):
        validate_inputs(None, None)

    with pytest.raises(ValueError, match="Training data cannot be empty."):
        validate_inputs(np.array([]), np.array([]))

    with pytest.raises(ValueError, match="Features and labels must have the same number of samples."):
        validate_inputs(np.random.rand(10, 20), np.random.rand(5))


def test_create_pipeline():
    """
    Test create_pipeline function.
    """
    model = RandomForestClassifier()
    pipeline = create_pipeline(model, n_components=10)
    assert isinstance(pipeline, Pipeline)
    assert 'pca' in pipeline.named_steps
    assert 'randomforestclassifier' in pipeline.named_steps


def test_perform_grid_search(sample_data):
    """
    Test perform_grid_search with Random Forest.
    """
    X_train, y_train = sample_data
    pipeline = create_pipeline(RandomForestClassifier(random_state=42), n_components=5)
    param_grid = {
        'randomforestclassifier__n_estimators': [10],
        'randomforestclassifier__max_depth': [None],
    }

    best_estimator, grid_search, _ = perform_grid_search(pipeline, param_grid, X_train, y_train)

    assert grid_search.best_params_ is not None
    assert grid_search.best_estimator_ == best_estimator


def test_train_random_forest(sample_data):
    """
    Test train_random_forest function.
    """
    X_train, y_train = sample_data
    best_model, _, _ = train_random_forest(X_train, y_train, n_components=5)
    assert hasattr(best_model, 'predict')  # Ensure the returned model has a predict method


def test_train_logistic_regression(sample_data):
    """
    Test train_logistic_regression function.
    """
    X_train, y_train = sample_data
    best_model, _, _ = train_logistic_regression(X_train, y_train, n_components=5)
    assert hasattr(best_model, 'predict')


def test_train_svm(sample_data):
    """
    Test train_svm function.
    """
    X_train, y_train = sample_data
    best_model, _, _ = train_svm(X_train, y_train, n_components=5)
    assert hasattr(best_model, 'predict')


def test_train_k_neighbors(sample_data):
    """
    Test train_k_neighbors function.
    """
    X_train, y_train = sample_data
    best_model, _, _ = train_k_neighbors(X_train, y_train, n_components=5)
    assert hasattr(best_model, 'predict')
