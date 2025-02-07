"""
Machine Learning Model Training with PCA and Hyperparameter Tuning

This module provides functions to train machine learning models using PCA
for dimensionality reduction and GridSearchCV for hyperparameter tuning.

Supported models:
- Random Forest
- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

Key Features:
- Modularized pipeline creation and training process
- Centralized parameter grids for easy maintainability
- Comprehensive logging for training and evaluation
- Support for hyperparameter tuning with GridSearchCV
"""

import logging
import time
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import matthews_corrcoef, make_scorer

# Centralized parameter grids for models
PARAM_GRIDS = {
    'random_forest': {
        'randomforestclassifier__n_estimators': [100, 200],
        'randomforestclassifier__max_depth': [None, 10, 20],
        'randomforestclassifier__min_samples_split': [2, 5],
        'randomforestclassifier__min_samples_leaf': [1, 2]
    },
    'svm': {
        'svc__C': [0.1, 1, 10],
        'svc__gamma': ['scale', 'auto']
    },
    'k_neighbors': {
        'kneighborsclassifier__n_neighbors': [3, 5, 7],
        'kneighborsclassifier__weights': ['uniform', 'distance'],
        'kneighborsclassifier__metric': ['euclidean', 'manhattan']
    },
    'logistic_regression': {
        'logisticregression__C': [0.1, 1, 10],
        'logisticregression__penalty': ['l2']
    }
}


def validate_inputs(X_train, y_train):
    """
    Validate training data inputs.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.

    Raises:
        ValueError: If inputs are invalid.
    """
    if X_train is None or y_train is None:
        raise ValueError("Training data cannot be None.")
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Training data cannot be empty.")
    if len(X_train) != len(y_train):
        raise ValueError("Features and labels must have the same number of samples.")


def create_pipeline(model, n_components=50):
    """
    Create a pipeline with PCA and the specified model.

    Args:
        model (sklearn.base.BaseEstimator): The model to include in the pipeline.
        n_components (int): Number of PCA components.

    Returns:
        sklearn.pipeline.Pipeline: The created pipeline.
    """
    return Pipeline([
        ('pca', PCA(n_components=n_components, random_state=42)),
        (model.__class__.__name__.lower(), model)
    ])


def perform_grid_search(pipeline, param_grid, X_train, y_train):
    """
    Perform grid search with cross-validation to find the best hyperparameters.

    Args:
        pipeline (sklearn.pipeline.Pipeline): The pipeline to use for grid search.
        param_grid (dict): The parameter grid for grid search.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.

    Returns:
        tuple: Best estimator, grid search object, and parameter grid.
    """
    validate_inputs(X_train, y_train)

    start_time = time.time()
    mcc_scorer = make_scorer(matthews_corrcoef)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring=mcc_scorer,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    end_time = time.time()

    logging.info(f"Grid search completed in {end_time - start_time:.2f} seconds")
    logging.info(f"Best parameters: {grid_search.best_params_}")
    logging.info(f"Best cross-validation MCC: {grid_search.best_score_:.2f}")

    return grid_search.best_estimator_, grid_search, param_grid


def train_model(model, param_grid, X_train, y_train, n_components=50):
    """
    Train a model with PCA and grid search for hyperparameter tuning.

    Args:
        model (sklearn.base.BaseEstimator): The model to train.
        param_grid (dict): The parameter grid for hyperparameter tuning.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        n_components (int): Number of PCA components.

    Returns:
        tuple: Best estimator, grid search object, and parameter grid.
    """
    pipeline = create_pipeline(model, n_components)
    return perform_grid_search(pipeline, param_grid, X_train, y_train)


def train_random_forest(X_train, y_train, n_components=50):
    return train_model(
        RandomForestClassifier(class_weight='balanced', random_state=42),
        PARAM_GRIDS['random_forest'], X_train, y_train, n_components
    )


def train_logistic_regression(X_train, y_train, n_components=50):
    return train_model(
        LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        PARAM_GRIDS['logistic_regression'], X_train, y_train, n_components
    )


def train_svm(X_train, y_train, n_components=50):
    return train_model(
        SVC(kernel='linear', class_weight='balanced', probability=True, random_state=42),
        PARAM_GRIDS['svm'], X_train, y_train, n_components
    )


def train_k_neighbors(X_train, y_train, n_components=50):
    return train_model(
        KNeighborsClassifier(),
        PARAM_GRIDS['k_neighbors'], X_train, y_train, n_components
    )
