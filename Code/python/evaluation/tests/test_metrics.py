import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from evaluation.metrics import (
    calculate_evaluation_metrics, combine_mccs,
    decode_labels, ensure_binary_classification, determine_positive_label
)


@pytest.fixture
def sample_labels():
    true_labels = np.array(['Vertebrate', 'Non-Vertebrate', 'Vertebrate', 'Non-Vertebrate'])
    predicted_labels = np.array(['Vertebrate', 'Vertebrate', 'Non-Vertebrate', 'Non-Vertebrate'])
    return true_labels, predicted_labels

@pytest.fixture
def label_encoder():
    le = LabelEncoder()
    le.fit(['Non-Vertebrate', 'Vertebrate'])
    return le

def test_calculate_evaluation_metrics(sample_labels, label_encoder):
    true_labels, predicted_labels = sample_labels
    metrics = calculate_evaluation_metrics(true_labels, predicted_labels, label_encoder=label_encoder)
    assert 'Accuracy' in metrics
    assert 'MCC' in metrics
    assert 'Precision' in metrics
    assert 'Recall' in metrics
    assert 'F1-Score' in metrics
    assert 'Per-Type MCC' in metrics

def test_combine_mccs():
    df1 = pd.DataFrame({'Type': ['A', 'B'], 'MCC1': [0.8, 0.6]})
    df2 = pd.DataFrame({'Type': ['A', 'B'], 'MCC2': [0.7, 0.5]})
    combined_df = combine_mccs(df1, df2)
    assert 'MCC1' in combined_df.columns
    assert 'MCC2' in combined_df.columns
    assert combined_df.shape == (2, 3)

def test_calculate_evaluation_metrics_empty_labels(label_encoder):
    with pytest.raises(ValueError, match="True labels array is empty."):
        calculate_evaluation_metrics([], [], label_encoder=label_encoder)

def test_calculate_evaluation_metrics_mismatched_lengths(label_encoder):
    with pytest.raises(ValueError, match="True labels and predicted labels must have the same number of samples."):
        calculate_evaluation_metrics(['Vertebrate'], ['Non-Vertebrate', 'Vertebrate'], label_encoder=label_encoder)

def test_calculate_evaluation_metrics_non_binary_classification(label_encoder):
    true_labels = ['A', 'B', 'C']
    predicted_labels = ['A', 'B', 'C']
    with pytest.raises(ValueError, match="Expected 2 unique labels for binary classification"):
        calculate_evaluation_metrics(true_labels, predicted_labels, label_encoder=label_encoder)

def test_combine_mccs_empty_dataframes():
    df1 = pd.DataFrame(columns=['Type', 'MCC1'])
    df2 = pd.DataFrame(columns=['Type', 'MCC2'])
    combined_df = combine_mccs(df1, df2)
    assert combined_df.empty

def test_combine_mccs_missing_type_column():
    df1 = pd.DataFrame({'MCC1': [0.8, 0.6]})
    df2 = pd.DataFrame({'MCC2': [0.7, 0.5]})
    with pytest.raises(KeyError, match="'Type'"):
        combine_mccs(df1, df2)

def test_decode_labels(label_encoder):
    true_labels = np.array([0, 1])
    predicted_labels = np.array([1, 0])
    decoded_true, decoded_predicted = decode_labels(true_labels, predicted_labels, label_encoder)
    assert np.array_equal(decoded_true, ['Non-Vertebrate', 'Vertebrate'])
    assert np.array_equal(decoded_predicted, ['Vertebrate', 'Non-Vertebrate'])

def test_ensure_binary_classification():
    with pytest.raises(ValueError, match="Expected 2 unique labels for binary classification"):
        ensure_binary_classification(['A', 'B', 'C'])

def test_combine_mccs_values():
    df1 = pd.DataFrame({'Type': ['A', 'B'], 'MCC1': [0.8, 0.6]})
    df2 = pd.DataFrame({'Type': ['A', 'B'], 'MCC2': [0.7, 0.5]})
    combined_df = combine_mccs(df1, df2)
    assert combined_df.loc[combined_df['Type'] == 'A', 'MCC1'].values[0] == 0.8
    assert combined_df.loc[combined_df['Type'] == 'A', 'MCC2'].values[0] == 0.7

def test_determine_positive_label():
    true_labels = np.array(['A', 'B'])
    pos_label = determine_positive_label(true_labels, 'LabelType', preferred_positive='C')
    assert pos_label == 'B'  # Should default to the second unique label
