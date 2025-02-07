import logging
import os
import pandas as pd
import h5py

def load_embeddings(csv_path):
    """
    Loads embeddings from a CSV (or other standard format).
    Modify based on your actual file format.
    """
    try:
        df = pd.read_csv(csv_path)
        logging.info(f"Loaded embeddings from {csv_path} with shape {df.shape}.")
        return df
    except Exception as e:
        logging.error(f"Error loading embeddings: {e}")
        return None

import h5py
import pandas as pd

def load_folds_based_embeddings(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            embeddings_dict = {
                protein_id: f[protein_id][:]
                for protein_id in f.keys()
            }

        # Quick debug: Are all entries identical?
        # Check the shape of each array from the dictionary
        for pid, arr in embeddings_dict.items():
            print(pid, arr.shape, arr.mean(), arr.std())
            # Optionally break after a few so you don't spam the terminal
            break  

        embeddings_df = pd.DataFrame.from_dict(embeddings_dict, orient='index')
        embeddings_df.index.name = 'ID'
        return embeddings_df.reset_index()

    except Exception as e:
        logging.error(f"Error loading folds-based embeddings: {e}")
        return None




def load_labels(csv_file_path, selected_column: str = 'target'):
    """
    Load labels from a CSV file and return a DataFrame with 'ID' and selected_column columns.

    Args:
        csv_file_path (str): Path to the CSV file containing labels.

    Returns:
        pd.DataFrame: DataFrame containing labels with 'ID' and selected_column columns.
    """
    try:
        labels_df = pd.read_csv(csv_file_path, delimiter=';')  # or ','
        print(labels_df.columns)  

        labels_df['ID'] = labels_df['ID'].str.replace('_', '.')
        logging.info(f"Loaded {len(labels_df)} labels from {csv_file_path}.")
        logging.info(f"{labels_df[['ID', selected_column]]}")
        return labels_df[['ID', selected_column]]


    except Exception as e:
        logging.error(f"Error loading labels from {csv_file_path}: {e}")
        return None

def merge_embeddings_labels(embeddings_df, labels_df):
    try:
        merged_df = pd.merge(embeddings_df, labels_df, on='ID')
        logging.info(f"Merged embeddings and labels. Result shape: {merged_df.shape}.")
        print("Embeddings IDs sample:", embeddings_df['ID'].head())
        print("Labels IDs sample:", labels_df['ID'].head())

        # Check how many overlap
        common_ids = set(embeddings_df['ID']).intersection(set(labels_df['ID']))
        print("Number of common IDs:", len(common_ids))
        print("A few common IDs:", list(common_ids)[:10])
        # After you load embeddings_df and labels_df but before merging:

        embeddings_ids = set(embeddings_df['ID'])
        labels_ids = set(labels_df['ID'])

        common_ids = embeddings_ids.intersection(labels_ids)
        only_in_embeddings = embeddings_ids.difference(labels_ids)
        only_in_labels = labels_ids.difference(embeddings_ids)

        print("Number of IDs in embeddings:", len(embeddings_ids))
        print("Number of IDs in labels:", len(labels_ids))
        print("Number of common IDs:", len(common_ids))
        print("\n=== Common IDs ===")
        for cid in sorted(common_ids):
            print(cid)

        print("\n=== IDs only in embeddings ===")
        for eid in sorted(only_in_embeddings):
            print(eid)

        print("\n=== IDs only in labels ===")
        for lid in sorted(only_in_labels):
            print(lid)
        
        print(merged_df)
        return merged_df
    except Exception as e:
        logging.error(f"Error merging embeddings and labels: {e}")
        return None


