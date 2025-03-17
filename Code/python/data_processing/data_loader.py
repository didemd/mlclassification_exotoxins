import logging
import numpy as np
import pandas as pd
import h5py

def load_embeddings(embeddings_file_path):
    """
    Load embeddings from an HDF5 file and return as a pandas DataFrame.

    Args:
        embeddings_file_path (str): Path to the HDF5 file containing embeddings.

    Returns:
        pd.DataFrame: DataFrame containing embeddings with 'ID' and feature columns.
    """
    try:
        embeddings_dict = _read_hdf5_file(embeddings_file_path)
        embeddings_df = _convert_dict_to_dataframe(embeddings_dict)
        embeddings_df = _preprocess_id_column(embeddings_df)
        logging.info(f"Loaded {len(embeddings_df)} embeddings with shape {embeddings_df.shape}.")
        return embeddings_df
    except Exception as e:
        logging.error(f"Error loading embeddings: {e}")
        return None

def _read_hdf5_file(embeddings_file_path):
    """
    Read embeddings from an HDF5 file into a dictionary.

    Args:
        embeddings_file_path (str): Path to the HDF5 file.

    Returns:
        dict: Dictionary containing embeddings.
    """
    embeddings_dict = {}
    with h5py.File(embeddings_file_path, 'r') as f:
        logging.info(f"Opened HDF5 file: {embeddings_file_path}")
        for key in f.keys():
            emb_array = np.array(f[key])
            if emb_array.ndim == 1:
                embeddings_dict[key.replace('.', '_')] = emb_array
            elif emb_array.ndim == 2:
                for idx, emb in enumerate(emb_array):
                    unique_key = f"{key.replace('.', '_')}_{idx}"
                    embeddings_dict[unique_key] = emb
            else:
                logging.warning(f"Embedding for key {key} has unsupported dimensions: {emb_array.ndim}")
    return embeddings_dict

def _convert_dict_to_dataframe(embeddings_dict):
    """
    Convert a dictionary of embeddings to a pandas DataFrame.

    Args:
        embeddings_dict (dict): Dictionary containing embeddings.

    Returns:
        pd.DataFrame: DataFrame containing embeddings.
    """
    embeddings_df = pd.DataFrame.from_dict(embeddings_dict, orient='index')
    embeddings_df.reset_index(inplace=True)
    embeddings_df.rename(columns={'index': 'ID'}, inplace=True)
    embeddings_df.columns = ['ID'] + [f'feature_{i}' for i in range(embeddings_df.shape[1] - 1)]
    return embeddings_df

def _preprocess_id_column(embeddings_df):
    """
    Preprocess the 'ID' column to retain only the ID number before any '|' character.

    Args:
        embeddings_df (pd.DataFrame): DataFrame containing embeddings.

    Returns:
        pd.DataFrame: DataFrame with preprocessed 'ID' column.
    """
    embeddings_df['ID'] = embeddings_df['ID'].apply(lambda x: x.split('|')[0].strip())
    logging.info("Preprocessed 'ID' column to retain only the ID number.")
    return embeddings_df

def load_labels(csv_file_path, selected_column: str = 'type'):
    """
    Load labels from a CSV file and return a DataFrame with 'ID' and selected_column columns.

    Args:
        csv_file_path (str): Path to the CSV file containing labels.

    Returns:
        pd.DataFrame: DataFrame containing labels with 'ID' and selected_column columns.
    """
    try:
        labels_df = pd.read_csv(csv_file_path, delimiter=';')
        labels_df['ID'] = labels_df['ID'].str.replace('.', '_')
        logging.info(f"Loaded {len(labels_df)} labels from {csv_file_path}.")
        logging.info(f"{labels_df[['ID', selected_column]]}")
        print(labels_df['type'].dtype)
        print(labels_df['type'].unique())
        return labels_df[['ID', selected_column]]
    except Exception as e:
        logging.error(f"Error loading labels from {csv_file_path}: {e}")
        return None

def merge_embeddings_labels(embeddings_df, labels_df, selected_column: str = 'type'):
    """
    Merge embeddings and labels DataFrames on 'ID' and remove duplicates.

    Args:
        embeddings_df (pd.DataFrame): DataFrame containing embeddings.
        labels_df (pd.DataFrame): DataFrame containing labels.

    Returns:
        pd.DataFrame: Merged DataFrame containing embeddings and labels.
    """
    merged_df = pd.merge(embeddings_df, labels_df, on='ID', how='inner')

    if selected_column == "type":
        merged_df['type'] = merged_df['type'].replace({
            'TypeIII_controlToxin': 'Type_III',
            'TypeII_PFT_bacteria': 'Type_II',
            'TypeIII' : 'Type_III'
        })
        logging.info(f"Merged DataFrame contains {len(merged_df)} samples after processing 'type' labels.")

    logging.info(f"Merged DataFrame contains {len(merged_df)} samples.")
    merged_df = _remove_duplicates(merged_df)
    print(merged_df)
    return merged_df

def _remove_duplicates(merged_df):
    """
    Remove duplicate rows based on the 'ID' column.

    Args:
        merged_df (pd.DataFrame): Merged DataFrame containing embeddings and labels.

    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    duplicates = merged_df.duplicated(subset='ID').sum()
    logging.info(f"Number of duplicate IDs: {duplicates}")
    if duplicates > 0:
        logging.warning(f"Removing {duplicates} duplicate IDs.")
        merged_df = merged_df.drop_duplicates(subset='ID')
        logging.info(f"DataFrame shape after removing duplicates: {merged_df.shape}")
    return merged_df
