import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
from evaluation.confusion_matrix_plot import compute_confusion_matrix, decode_labels
from unittest.mock import patch

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

def test_decode_labels():
    y_test = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])
    label_encoder = LabelEncoder()
    label_encoder.fit(['Non-Vertebrate', 'Vertebrate'])
    y_test_decoded, y_pred_decoded = decode_labels(y_test, y_pred, label_encoder)
    assert all(y_test_decoded == ['Non-Vertebrate', 'Vertebrate', 'Non-Vertebrate', 'Vertebrate'])
    assert all(y_pred_decoded == ['Non-Vertebrate', 'Vertebrate', 'Vertebrate', 'Non-Vertebrate'])

def test_decode_labels_without_encoder():
    y_test = np.array(['Non-Vertebrate', 'Vertebrate', 'Non-Vertebrate', 'Vertebrate'])
    y_pred = np.array(['Non-Vertebrate', 'Vertebrate', 'Vertebrate', 'Non-Vertebrate'])
    y_test_decoded, y_pred_decoded = decode_labels(y_test, y_pred, None)
    assert all(y_test_decoded == y_test)
    assert all(y_pred_decoded == y_pred)

@patch('evaluation.confusion_matrix_plot.plot_confusion_matrix')
@patch('evaluation.confusion_matrix_plot.save_plot')
def test_compute_confusion_matrix(mock_save_plot, mock_plot_confusion_matrix, trained_model, sample_data):
    X, y = sample_data
    labels = ['Non-Vertebrate', 'Vertebrate']
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    compute_confusion_matrix(trained_model, X, y, labels, 'test_dataset', 'test_predictor', 'RandomForest', label_encoder)
    
    # Check if the plot_confusion_matrix and save_plot functions were called
    assert mock_plot_confusion_matrix.called
    assert mock_save_plot.called
