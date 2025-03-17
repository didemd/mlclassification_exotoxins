import logging
import numpy as np
import pandas as pd
from config.config import PLOT_SAVE_DIR
"""
def load_blast_hits(blast_results_path='./data/blast_results.tsv'):

    try:
        top_hits_df = pd.read_csv(
            blast_results_path,
            sep='\t',
            header=None,
            names=[
                'qseqid','sseqid','pident','length','mismatch','gapopen',
                'qstart','qend','sstart','send','evalue','bitscore'
            ],
            comment='#',
            dtype={
                'qseqid': str,
                'sseqid': str,
                'pident': float,
                'length': int,
                'mismatch': int,
                'gapopen': int,
                'qstart': int,
                'qend': int,
                'sstart': int,
                'send': int,
                'evalue': float,
                'bitscore': float
            },
            engine='python'
        )

        top_hits_df['sseqid'] = top_hits_df['sseqid'].str.strip().str.upper()
        top_hits_df.columns = top_hits_df.columns.str.strip()
        logging.info(f"Loaded BLAST hits from {blast_results_path}")
        return top_hits_df
    except FileNotFoundError:
        logging.error(f"BLAST results file not found at path: {blast_results_path}")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        logging.error(f"BLAST results file at {blast_results_path} is empty.")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error loading BLAST results: {e}")
        return pd.DataFrame()


def load_blast_labels(labels_file_path='./data/ToxinTypes_labelTarget_3.csv'):

    try:
        train_labels_df = pd.read_csv(
            labels_file_path,
            delimiter=';',
            header=0,
            names=['ID', 'species', 'type', 'target'],
            skiprows=1
        )
    except FileNotFoundError:
        logging.error(f"Labels file not found at path: {labels_file_path}")
        return {}, {}
    except Exception as e:
        logging.error(f"Error loading labels from {labels_file_path}: {e}")
        return {}, {}

    type_mapping = {
        'TypeIII': 'Type_III',
        'TypeIII_controlToxin': 'Type_III',
        'TypeII_PFT_bacteria': 'Type_II',
        'Type_III': 'Type_III',
        'Type II': 'Type_II',
        'Type III': 'Type_III',
        'Type I': 'Type_I',
        'Type IV': 'Type_IV'
    }

    train_labels_df['type'] = train_labels_df['type'].replace(type_mapping)
    train_labels_df['ID'] = train_labels_df['ID'].str.strip().str.upper()

    type_map = pd.Series(
        train_labels_df.type.values,
        index=train_labels_df.ID
    ).to_dict()

    target_map = pd.Series(
        train_labels_df.target.values,
        index=train_labels_df.ID
    ).to_dict()

    # Standardize
    type_map = {k.strip().upper(): v for k, v in type_map.items()}
    target_map = {k.strip().upper(): v for k, v in target_map.items()}

    return type_map, target_map


def run_blast_predictor(top_hits_df, type_map, target_map):

    required_columns = ['qseqid', 'evalue']
    for col in required_columns:
        if col not in top_hits_df.columns:
            logging.error(f"Missing required column '{col}' in BLAST hits DataFrame.")
            return pd.DataFrame()

    # Sort by qseqid and evalue (ascending: lowest evalue first)
    top_hits_df_sorted = top_hits_df.sort_values(['qseqid', 'evalue'], ascending=[True, True])

    # Keep the top hit per qseqid and make a copy to avoid SettingWithCopyWarning
    top_hits_unique = top_hits_df_sorted.drop_duplicates(subset='qseqid', keep='first').copy()

    # Map labels using .loc to ensure we're modifying the copy
    top_hits_unique.loc[:, 'predicted_exotoxin_type'] = top_hits_unique['sseqid'].map(type_map)
    top_hits_unique.loc[:, 'predicted_target'] = top_hits_unique['sseqid'].map(target_map)

    # Replace NaN with 'Unknown'
    top_hits_unique.loc[:, 'predicted_exotoxin_type'] = top_hits_unique['predicted_exotoxin_type'].fillna('Unknown')
    top_hits_unique.loc[:, 'predicted_target'] = top_hits_unique['predicted_target'].fillna('Unknown')

    logging.info(f"Selected top BLAST hits per query: {top_hits_unique.shape[0]} records")
    return top_hits_unique

# evaluation/confusion_matrix_plot.py
"""
import logging
import numpy as np
import pandas as pd
from config.config import PLOT_SAVE_DIR
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('Logs/blast_predictor.log')
    ]
)

def load_blast_hits(blast_results_path='./data/blast_results.tsv'):
    """
    Load BLAST hits from a TSV file.
    """
    try:
        top_hits_df = pd.read_csv(
            blast_results_path,
            sep='\t',
            header=None,
            names=[
                'qseqid','sseqid','pident','length','mismatch','gapopen',
                'qstart','qend','sstart','send','evalue','bitscore'
            ],
            comment='#',
            dtype={
                'qseqid': str,
                'sseqid': str,
                'pident': float,
                'length': int,
                'mismatch': int,
                'gapopen': int,
                'qstart': int,
                'qend': int,
                'sstart': int,
                'send': int,
                'evalue': float,
                'bitscore': float
            },
            engine='python'
        )

        top_hits_df['sseqid'] = top_hits_df['sseqid'].str.strip().str.upper()
        top_hits_df.columns = top_hits_df.columns.str.strip()
        logging.info(f"Loaded BLAST hits from {blast_results_path}")
        return top_hits_df
    except FileNotFoundError:
        logging.error(f"BLAST results file not found at path: {blast_results_path}")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        logging.error(f"BLAST results file at {blast_results_path} is empty.")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error loading BLAST results: {e}")
        return pd.DataFrame()


def load_blast_labels(labels_file_path='./data/ToxinTypes_labelTarget_3.csv'):
    """
    Load labels for BLAST references from a CSV file.
    """
    try:
        train_labels_df = pd.read_csv(
            labels_file_path,
            delimiter=';',
            header=0,
            names=['ID', 'species', 'type', 'target'],
            skiprows=1
        )
    except FileNotFoundError:
        logging.error(f"Labels file not found at path: {labels_file_path}")
        return {}, {}
    except Exception as e:
        logging.error(f"Error loading labels from {labels_file_path}: {e}")
        return {}, {}

    type_mapping = {
        'TypeIII': 'Type_III',
        'TypeIII_controlToxin': 'Type_III',
        'TypeII_PFT_bacteria': 'Type_II',
        'Type_III': 'Type_III',
        'Type II': 'Type_II',
        'Type III': 'Type_III',
        'Type I': 'Type_I',
        'Type IV': 'Type_IV'
    }

    train_labels_df['type'] = train_labels_df['type'].replace(type_mapping)
    train_labels_df['ID'] = train_labels_df['ID'].str.strip().str.upper()

    type_map = pd.Series(
        train_labels_df.type.values,
        index=train_labels_df.ID
    ).to_dict()

    target_map = pd.Series(
        train_labels_df.target.values,
        index=train_labels_df.ID
    ).to_dict()

    # Standardize
    type_map = {k.strip().upper(): v for k, v in type_map.items()}
    target_map = {k.strip().upper(): v for k, v in target_map.items()}

    logging.info(f"Loaded label mappings from {labels_file_path}")
    return type_map, target_map


def run_blast_predictor(top_hits_df, type_map, target_map):
    """
    Map BLAST top hits to predicted type and target by selecting the top hit (lowest e-value) per qseqid.
    Additionally, prepare features and labels for ML model training.
    """
    required_columns = ['qseqid', 'evalue']
    for col in required_columns:
        if col not in top_hits_df.columns:
            logging.error(f"Missing required column '{col}' in BLAST hits DataFrame.")
            return pd.DataFrame(), None, None

    # Sort by qseqid and evalue (ascending: lowest evalue first)
    top_hits_df_sorted = top_hits_df.sort_values(['qseqid', 'evalue'], ascending=[True, True])

    # Keep the top hit per qseqid and make a copy to avoid SettingWithCopyWarning
    top_hits_unique = top_hits_df_sorted.drop_duplicates(subset='qseqid', keep='first').copy()

    # Map labels using .loc to ensure we're modifying the copy
    top_hits_unique.loc[:, 'predicted_exotoxin_type'] = top_hits_unique['sseqid'].map(type_map)
    top_hits_unique.loc[:, 'predicted_target'] = top_hits_unique['sseqid'].map(target_map)

    # Replace NaN with 'Unknown'
    top_hits_unique['predicted_exotoxin_type'] = top_hits_unique['predicted_exotoxin_type'].fillna('Unknown')
    top_hits_unique['predicted_target'] = top_hits_unique['predicted_target'].fillna('Unknown')

    logging.info(f"Selected top BLAST hits per query: {top_hits_unique.shape[0]} records")
    return top_hits_unique



def extract_features_and_labels(predicted_hits, label_type='predicted_exotoxin_type'):
    """
    Extract features and labels from the predicted hits DataFrame for ML model training.
    
    Parameters:
    - predicted_hits: DataFrame with BLAST hits and mapped labels.
    - label_type: The type of label to predict ('predicted_exotoxin_type' or 'predicted_target').
    
    Returns:
    - X: Feature matrix.
    - y: Labels.
    """
    # Define feature columns
    feature_columns = ['pident', 'length', 'mismatch', 'gapopen', 'qstart',
                       'qend', 'sstart', 'send', 'evalue', 'bitscore']

    # Ensure all feature columns are present
    for col in feature_columns:
        if col not in predicted_hits.columns:
            logging.error(f"Missing feature column '{col}' in DataFrame.")
            return None, None

    X = predicted_hits[feature_columns]
    y = predicted_hits[label_type]

    return X, y




def preprocess_features(X):
    """
    Preprocess feature matrix X. This can include normalization, encoding, etc.
    For simplicity, we'll perform basic scaling on numerical features.
    
    Parameters:
    - X: Feature matrix.
    
    Returns:
    - X_scaled: Scaled feature matrix.
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler for future use
    scaler_path = os.path.join(PLOT_SAVE_DIR, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    logging.info(f"Saved scaler to {scaler_path}")

    return X_scaled


def train_model(X_train, y_train, model_path='./models/random_forest_type.pkl'):
    """
    Train a Random Forest classifier.
    
    Parameters:
    - X_train: Training features.
    - y_train: Training labels.
    - model_path: Path to save the trained model.
    
    Returns:
    - model: Trained Random Forest model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    logging.info(f"Trained Random Forest model saved to {model_path}")
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.
    
    Parameters:
    - model: Trained ML model.
    - X_test: Testing features.
    - y_test: Testing labels.
    
    Returns:
    - None (prints evaluation metrics)
    """
    y_pred = model.predict(X_test)
    logging.info("Classification Report:")
    logging.info("\n" + classification_report(y_test, y_pred))
    logging.info("Confusion Matrix:")
    logging.info("\n" + str(confusion_matrix(y_test, y_pred)))



def compute_confusion_matrix_blast(cm, labels, dataset_type, mode, predictor_name):
    """
    Compute and plot confusion matrix for BLAST predictor.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix for {predictor_name} on {dataset_type} Data')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_SAVE_DIR, f"{predictor_name}_confusion_matrix.png"))
    plt.close()
    logging.info(f"Confusion matrix for {predictor_name} saved.")
