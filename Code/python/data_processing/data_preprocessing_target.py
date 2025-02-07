"""
This module provides functions for preprocessing data for a binary classification task.
The preprocessing steps include dropping missing values, normalizing labels, filtering
to binary classes, splitting features and labels, encoding labels, and logging various
statistics about the data.

Pipeline for preprocessing:

1- Drop missing values
2- Normalize labels
3- Log class distribution
4- Filter to two classes
5- Log updated distribution, check for valid classes
6- Split into X and y
7- Encode y, ensuring 'Vertebrate' is 1
8- Log final distribution & feature stats
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def drop_na(merged_df):
    """
    Drop rows with missing values from the DataFrame.

    Args:
        merged_df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame after dropping rows with missing values.
    """
    if not isinstance(merged_df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    merged_df = merged_df.dropna()
    logging.info(f"DataFrame shape after dropping NAs: {merged_df.shape}")
    return merged_df

def normalize_labels(merged_df):
    """
    Normalize the 'target' column by stripping whitespace and converting to title case.

    Args:
        merged_df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with normalized 'target' labels.
    """
    if 'target' not in merged_df.columns:
        raise ValueError("DataFrame must contain a 'target' column.")
    
    merged_df.loc[:, 'target'] = merged_df['target'].str.strip().str.title()
    return merged_df

def log_class_distribution(merged_df, message):
    """
    Log the distribution of classes in the 'target' column.

    Args:
        merged_df (pd.DataFrame): The input DataFrame.
        message (str): A message to log before the class distribution.
    """
    if 'target' not in merged_df.columns:
        raise ValueError("DataFrame must contain a 'target' column.")
    
    unique, counts = np.unique(merged_df['target'], return_counts=True)
    logging.info(message)
    for cls, cnt in zip(unique, counts):
        logging.info(f"{cls}: {cnt}")

def filter_binary_classes(merged_df):
    """
    Filter the DataFrame to include only rows with 'target' values of 'Non-Vertebrate' or 'Vertebrate'.

    Args:
        merged_df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame filtered to binary classes.
    """
    if 'target' not in merged_df.columns:
        raise ValueError("DataFrame must contain a 'target' column.")
    
    binary_classes = ['Non-Vertebrate', 'Vertebrate']
    merged_df = merged_df[merged_df['target'].isin(binary_classes)]
    logging.info(f"DataFrame shape after filtering to binary classes: {merged_df.shape}")
    return merged_df

def split_features_labels(merged_df):
    """
    Split the DataFrame into feature matrix X and label vector y.

    Args:
        merged_df (pd.DataFrame): The input DataFrame.

    Returns:
        tuple: A tuple containing the feature matrix X and label vector y.
    """
    if 'ID' not in merged_df.columns or 'target' not in merged_df.columns:
        raise ValueError("DataFrame must contain 'ID' and 'target' columns.")
    
    feature_columns = merged_df.drop(['ID', 'target'], axis=1).columns
    X = merged_df[feature_columns].values
    y = merged_df['target'].values
    logging.info(f"Feature matrix shape: {X.shape}")
    return X, y

def encode_labels(y):
    """
    Encode the labels using LabelEncoder, ensuring 'Vertebrate' is the positive class (1).

    Args:
        y (np.ndarray): The label vector.

    Returns:
        tuple: A tuple containing the encoded labels and the LabelEncoder instance.
    """
    if not isinstance(y, np.ndarray):
        raise ValueError("Input labels must be a numpy array.")
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    logging.info(f"Label Encoding: {label_mapping}")
    if label_mapping.get('Vertebrate') != 1:
        logging.warning("Adjusting label encoding to ensure 'Vertebrate' is the positive class (1).")
        y_encoded = 1 - y_encoded
        label_encoder.classes_ = np.array(['Non-Vertebrate', 'Vertebrate'])
    return y_encoded, label_encoder

def log_final_distribution(y_encoded, label_encoder):
    """
    Log the final distribution of encoded labels.

    Args:
        y_encoded (np.ndarray): The encoded label vector.
        label_encoder (LabelEncoder): The LabelEncoder instance used for encoding.
    """
    unique_final, counts_final = np.unique(y_encoded, return_counts=True)
    for cls, cnt in zip(label_encoder.inverse_transform(unique_final), counts_final):
        logging.info(f"{cls}: {cnt}")

def log_feature_statistics(merged_df):
    """
    Log statistical summary of the feature columns.

    Args:
        merged_df (pd.DataFrame): The input DataFrame.
    """
    feature_df = merged_df.drop(['ID', 'target'], axis=1)
    logging.info("Feature Statistics:")
    logging.info(feature_df.describe().transpose())

def preprocess_data(merged_df):
    """
    Preprocess the data by performing a series of steps including dropping NAs, normalizing labels,
    logging class distribution, filtering to binary classes, splitting features and labels, encoding labels,
    and logging final distribution and feature statistics.

    Args:
        merged_df (pd.DataFrame): The input DataFrame.

    Returns:
        tuple: A tuple containing the feature matrix X, encoded label vector y_encoded, and the LabelEncoder instance.
    """
    if not isinstance(merged_df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    merged_df = drop_na(merged_df)
    merged_df = normalize_labels(merged_df)
    log_class_distribution(merged_df, "Class Distribution Before Filtering:")
    merged_df = filter_binary_classes(merged_df)
    log_class_distribution(merged_df, "Class Distribution After Filtering:")
    
    if len(np.unique(merged_df['target'])) < 2:
        logging.error("Filtered data contains only one class.")
        raise ValueError("Filtered data must contain exactly two classes for binary classification.")
    
    X, y = split_features_labels(merged_df)
    y_encoded, label_encoder = encode_labels(y)
    log_final_distribution(y_encoded, label_encoder)
    log_feature_statistics(merged_df)
    logging.info("Data preprocessing completed.")
    
    return X, y_encoded, label_encoder
