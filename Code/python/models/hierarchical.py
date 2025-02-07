# my_project/models/hierarchical.py

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef

def prepare_first_level_labels(df):
    """
    Prepare first-level labels for hierarchical classification.
    Group1 = ['Type_I', 'Type_IV']
    Group2 = ['Type_II', 'Type_III']
    Group3 = ['Unknown'] (or other custom logic).
    """
    df = df.copy()
    df['first_level_label'] = df['type'].apply(
        lambda x: 'Group1' if x in ['Type_I', 'Type_IV'] else (
            'Group2' if x in ['Type_II', 'Type_III'] else (
                'Group3' if x == 'Unknown' else np.nan))
    )
    # Drop any rows that do not map to a group
    df.dropna(subset=['first_level_label'], inplace=True)
    logging.info(f"Prepared first-level labels. DataFrame now has {len(df)} samples.")

    return df

def train_second_level_classifiers_rf(df):
    """
    Train second-level RandomForest classifiers for each group.
    Returns a dictionary: { group_label: {'classifier': model, 'label_encoder': le}, ... }
    """
    second_level_classifiers = {}
    group_to_types = {
        'Group1': ['Type_I', 'Type_IV'],
        'Group2': ['Type_II', 'Type_III'],
        'Group3': ['Unknown']
    }

    for group, types in group_to_types.items():
        group_df = df[df['first_level_label'] == group]
        group_df = group_df[group_df['type'].isin(types)]

        if group_df.empty:
            logging.warning(f"No samples found for {group}. Skipping classifier training for this group.")
            continue

        feature_columns = [col for col in group_df.columns if col.startswith('feature_')]
        X_group = group_df[feature_columns].values
        y_group = group_df['type'].values

        group_label_encoder = LabelEncoder()
        y_group_encoded = group_label_encoder.fit_transform(y_group)

        # If only one class, no training needed
        if len(group_label_encoder.classes_) == 1:
            logging.info(f"Only one class in {group}. Skipping classifier training.")
            second_level_classifiers[group] = {
                'classifier': None,
                'label_encoder': group_label_encoder
            }
            continue

        clf = RandomForestClassifier(class_weight='balanced', random_state=42)
        clf.fit(X_group, y_group_encoded)

        y_pred = clf.predict(X_group)
        mcc = matthews_corrcoef(y_group_encoded, y_pred)
        logging.info(f"Second-level classifier MCC for {group} (RF): {mcc:.2f}")

        second_level_classifiers[group] = {
            'classifier': clf,
            'label_encoder': group_label_encoder
        }

    return second_level_classifiers


def train_second_level_classifiers_svm(df):
    """
    Train second-level SVM classifiers for each group.
    """
    from sklearn.svm import SVC

    second_level_classifiers = {}
    group_to_types = {
        'Group1': ['Type_I', 'Type_IV'],
        'Group2': ['Type_II', 'Type_III'],
        'Group3': ['Unknown']
    }

    for group, types in group_to_types.items():
        group_df = df[df['first_level_label'] == group]
        group_df = group_df[group_df['type'].isin(types)]

        if group_df.empty:
            logging.warning(f"No samples found for {group}. Skipping classifier training for this group.")
            continue

        feature_columns = [col for col in group_df.columns if col.startswith('feature_')]
        X_group = group_df[feature_columns].values
        y_group = group_df['type'].values

        group_label_encoder = LabelEncoder()
        y_group_encoded = group_label_encoder.fit_transform(y_group)

        if len(group_label_encoder.classes_) == 1:
            logging.info(f"Only one class in {group}. Skipping classifier training.")
            second_level_classifiers[group] = {
                'classifier': None,
                'label_encoder': group_label_encoder
            }
            continue

        clf = SVC(kernel='linear', class_weight='balanced', probability=True, random_state=42)
        clf.fit(X_group, y_group_encoded)

        y_pred = clf.predict(X_group)
        mcc = matthews_corrcoef(y_group_encoded, y_pred)
        logging.info(f"Second-level classifier MCC for {group} (SVM): {mcc:.2f}")

        second_level_classifiers[group] = {
            'classifier': clf,
            'label_encoder': group_label_encoder
        }

    return second_level_classifiers


class HierarchicalClassifier:
    """
    A hierarchical classifier that first predicts Group1/Group2/Group3, then 
    uses a second-level classifier to distinguish specific subtypes within each group.
    """
    def __init__(self, first_level_clf, second_level_classifiers, first_level_label_encoder):
        self.first_level_clf = first_level_clf
        self.second_level_classifiers = second_level_classifiers
        self.first_level_label_encoder = first_level_label_encoder

    def predict(self, X):
        first_level_pred_indices = self.first_level_clf.predict(X)
        first_level_labels = self.first_level_label_encoder.inverse_transform(first_level_pred_indices)

        final_predictions = []
        for i in range(len(X)):
            x_sample = X[i].reshape(1, -1)
            group_label = first_level_labels[i]

            clf_info = self.second_level_classifiers.get(group_label)
            if clf_info:
                clf = clf_info['classifier']
                le = clf_info['label_encoder']
                if clf is None:
                    # Only one class in this group
                    type_pred = le.inverse_transform([0])[0]
                else:
                    type_pred_encoded = clf.predict(x_sample)[0]
                    type_pred = le.inverse_transform([type_pred_encoded])[0]
                final_predictions.append(type_pred)
            else:
                logging.warning(f"Group '{group_label}' not recognized. Assigning 'Unknown'.")
                final_predictions.append('Unknown')

        return np.array(final_predictions)


def train_hierarchical_classifier_rf(training_df):
    """
    Train a hierarchical classifier (RandomForest at first level, RandomForest at second level).
    """
    from sklearn.ensemble import RandomForestClassifier

    prepared_df = prepare_first_level_labels(training_df)
    feature_columns = [col for col in prepared_df.columns if col.startswith('feature_')]
    X_train = prepared_df[feature_columns].values
    y_train = prepared_df['first_level_label'].values

    from sklearn.preprocessing import LabelEncoder
    first_level_label_encoder = LabelEncoder()
    y_encoded = first_level_label_encoder.fit_transform(y_train)

    first_level_clf = RandomForestClassifier(class_weight='balanced', random_state=42)
    first_level_clf.fit(X_train, y_encoded)

    # Overwrite the first-level label column with the actual encoded labels
    prepared_df['first_level_label'] = first_level_label_encoder.inverse_transform(y_encoded)

    second_level_classifiers_rf = train_second_level_classifiers_rf(prepared_df)
    model_rf = HierarchicalClassifier(first_level_clf, second_level_classifiers_rf, first_level_label_encoder)
    return model_rf


def train_hierarchical_classifier_svm(training_df):
    """
    Train a hierarchical classifier (SVM at first level, SVM at second level).
    """
    from sklearn.svm import SVC

    prepared_df = prepare_first_level_labels(training_df)
    feature_columns = [col for col in prepared_df.columns if col.startswith('feature_')]
    X_train = prepared_df[feature_columns].values
    y_train = prepared_df['first_level_label'].values

    from sklearn.preprocessing import LabelEncoder
    first_level_label_encoder = LabelEncoder()
    y_encoded = first_level_label_encoder.fit_transform(y_train)

    first_level_clf = SVC(kernel='linear', class_weight='balanced', probability=True, random_state=42)
    first_level_clf.fit(X_train, y_encoded)

    prepared_df['first_level_label'] = first_level_label_encoder.inverse_transform(y_encoded)

    second_level_classifiers_svm = train_second_level_classifiers_svm(prepared_df)
    model_svm = HierarchicalClassifier(first_level_clf, second_level_classifiers_svm, first_level_label_encoder)
    return model_svm
