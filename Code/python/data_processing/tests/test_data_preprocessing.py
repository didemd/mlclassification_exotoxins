import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from data_processing.data_preprocessing_target import (
    drop_na, normalize_labels, log_class_distribution, filter_binary_classes,
    split_features_labels, encode_labels, log_final_distribution, log_feature_statistics, preprocess_data
)


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'ID': ['1', '2', '3', '4'],
        'feature1': [1.0, 2.0, np.nan, 4.0],
        'feature2': [4.0, np.nan, 6.0, 8.0],
        'target': ['vertebrate', 'non-vertebrate', 'vertebrate', 'non-vertebrate']
    })

def test_drop_na(sample_dataframe):
    result = drop_na(sample_dataframe)
    assert result.shape[0] == 2

def test_normalize_labels(sample_dataframe):
    result = normalize_labels(sample_dataframe)
    assert all(result['target'].str.istitle())

def test_filter_binary_classes(sample_dataframe):
    result = filter_binary_classes(sample_dataframe)
    assert all(result['target'].isin(['Non-Vertebrate', 'Vertebrate']))

def test_split_features_labels(sample_dataframe):
    features, labels = split_features_labels(sample_dataframe)
    assert features.shape[1] == 2
    assert len(labels) == 4

def test_encode_labels():
    y = np.array(['Vertebrate', 'Non-Vertebrate', 'Vertebrate', 'Non-Vertebrate'])
    y_encoded, label_encoder = encode_labels(y)
    assert set(y_encoded) == {0, 1}
    assert label_encoder.classes_[1] == 'Vertebrate'

def test_preprocess_data(sample_dataframe):
    # Test normal behavior
    X, y_encoded, label_encoder = preprocess_data(sample_dataframe)
    
    # Check that the features matrix (X) has the correct number of columns
    assert X.shape[1] == 2  # Ensure only the feature columns remain
    # Check that the labels vector (y_encoded) has the same length as the input DataFrame after filtering
    assert len(y_encoded) == len(sample_dataframe.dropna())  # Dropped rows with NaN values

    # Assert that the encoded labels vector y_encoded has the correct length
    assert len(y_encoded) == len(sample_dataframe.dropna()), (
        f"Expected {len(sample_dataframe.dropna())} labels, but got {len(y_encoded)}"
    )

    # Assert that the LabelEncoder instance encodes and decodes correctly
    decoded_labels = label_encoder.inverse_transform(y_encoded)
    expected_labels = sample_dataframe.dropna()["target"].str.strip().str.title()
    assert all(decoded_labels == expected_labels), (
        "Decoded labels do not match the original processed labels."
    )

    # Test for ValueError when input is not a DataFrame
    with pytest.raises(ValueError, match="Input must be a pandas DataFrame."):
        preprocess_data("not_a_dataframe")

    # Test for ValueError when the data does not contain two classes
    with pytest.raises(ValueError, match="Filtered data must contain exactly two classes for binary classification."):
        # Provide a DataFrame with only one class after filtering
        invalid_df = sample_dataframe[sample_dataframe['target'] == 'vertebrate']
        preprocess_data(invalid_df)
