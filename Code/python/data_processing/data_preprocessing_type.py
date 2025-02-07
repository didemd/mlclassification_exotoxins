
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(merged_df):
    merged_df = merged_df.dropna()
    logging.info(f"DataFrame shape after dropping NAs: {merged_df.shape}")

    feature_columns = merged_df.drop(['ID', 'type'], axis=1).columns
    X = merged_df[feature_columns].values
    logging.info(f"Feature matrix shape: {X.shape}")
    y = merged_df['type'].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    unique, counts = np.unique(y_encoded, return_counts=True)
    class_distribution = dict(zip(label_encoder.classes_, counts))
    logging.info("Class Distribution:")
    for cls, cnt in class_distribution.items():
        logging.info(f"{cls}: {cnt}")

    duplicates = merged_df.duplicated(subset='ID').sum()
    logging.info(f"Number of duplicate IDs after merging: {duplicates}")

    logging.info("Feature Statistics:")
    feature_df = merged_df.drop(['ID', 'type'], axis=1)
    logging.info(feature_df.describe().transpose())
    logging.info("Data preprocessing completed.")

    return X, y_encoded, label_encoder


def prepare_first_level_labels(df):
    df = df.copy()
    df['first_level_label'] = df['type'].apply(
        lambda x: 'Group1' if x in ['Type_I', 'Type_IV'] else (
            'Group2' if x in ['Type_II', 'Type_III'] else (
                'Group3' if x == 'Unknown' else np.nan))
    )
    df = df.dropna(subset=['first_level_label'])
    logging.info(f"Prepared first-level labels. DataFrame now has {len(df)} samples.")
    return df

# --------------------- Classifier Training Functions ---------------------
def train_second_level_classifiers_rf(df, label_encoder):
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

        # If the group is 'Unknown', no need to train a classifier as it's a single class
        if len(group_label_encoder.classes_) == 1:
            logging.info(f"Only one class in {group}. Skipping classifier training.")
            second_level_classifiers[group] = {
                'classifier': None,  # No classifier needed
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

