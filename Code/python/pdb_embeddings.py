#!/usr/bin/env python3
"""
generate_protein_embeddings.py

This script loads protein sequences from a FASTA file, generates embeddings using the ProtT5 model,
and saves the embeddings to an HDF5 file.

Dependencies:
    - biopython
    - torch
    - transformers
    - h5py
    - tqdm

Usage:
    python generate_protein_embeddings.py --fasta path/to/input.fasta --output path/to/output.h5
"""

import os
import re
import argparse
from typing import Dict

from Bio import SeqIO

import torch
from transformers import T5Tokenizer, T5Model
import h5py

from tqdm import tqdm


def load_fasta(file_path: str) -> Dict[str, str]:
    """
    Load protein sequences from a FASTA file.

    Parameters:
        file_path (str): Path to the FASTA file.

    Returns:
        dict: A dictionary where keys are protein IDs and values are sequences.
    """
    sequences = {}
    print(f"Loading sequences from FASTA file: {file_path}")
    try:
        for record in SeqIO.parse(file_path, "fasta"):
            protein_id = record.id
            sequence = str(record.seq).upper()
            if sequence:  # Ensure sequence is not empty
                sequences[protein_id] = sequence
            else:
                print(f"Warning: Sequence for {protein_id} is empty. Skipping.")
    except Exception as e:
        print(f"Error reading FASTA file: {e}")
        raise e

    print(f"Total sequences loaded: {len(sequences)}")
    return sequences


def load_prot_t5_model(model_name: str = "Rostlab/prot_t5_xl_uniref50", use_gpu: bool = True):
    """
    Load the ProtT5 model and tokenizer.

    Parameters:
        model_name (str): The name of the pretrained ProtT5 model.
        use_gpu (bool): Whether to use GPU if available.

    Returns:
        tokenizer: Loaded T5 tokenizer.
        model: Loaded T5 model.
        device: Torch device (CPU or GPU).
    """
    print(f"Loading ProtT5 model and tokenizer: {model_name}")
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        model = T5Model.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}")
        raise e

    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        print("Model moved to GPU.")
    else:
        device = torch.device("cpu")
        print("Using CPU for computations.")

    model.eval()  # Set model to evaluation mode
    return tokenizer, model, device


def sanitize_hdf5_key(key: str) -> str:
    """
    Sanitize the HDF5 key to ensure it meets HDF5 naming requirements.
    Replace or remove invalid characters.

    Parameters:
        key (str): Original key.

    Returns:
        str: Sanitized key.
    """
    sanitized = re.sub(r'[^\w\-]', '_', key)
    return sanitized


def generate_embeddings_in_batches(
    sequences: Dict[str, str],
    tokenizer,
    model,
    device: torch.device,
    embedding_type: str = "per-sequence",
    batch_size: int = 8,
    max_length: int = 1024
) -> Dict[str, torch.Tensor]:
    """
    Generate embeddings for protein sequences in batches using ProtT5.

    Parameters:
        sequences (dict): {protein_id: sequence}
        tokenizer: Loaded T5 tokenizer.
        model: Loaded T5 model.
        device (torch.device): Device to perform computations on.
        embedding_type (str): 'per-sequence' (mean) or 'per-residue' (full hidden states).
        batch_size (int): Number of sequences per batch.
        max_length (int): Maximum token length for the tokenizer.

    Returns:
        dict: {protein_id: torch.Tensor of embeddings}
    """
    print(f"Generating embeddings with embedding_type='{embedding_type}' and batch_size={batch_size}")
    embeddings = {}
    items = list(sequences.items())
    total_batches = (len(items) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(items), batch_size), desc="Generating Embeddings", total=total_batches):
        batch = items[i:i + batch_size]
        protein_ids = [pid for (pid, _) in batch]
        sequences_batch = [ " ".join(seq) for (_, seq) in batch ]  # Space-separated for ProtT5

        # Tokenize the batch
        tokens = tokenizer(
            sequences_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )

        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = model(**tokens)

        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]

        for idx, protein_id in enumerate(protein_ids):
            if embedding_type == "per-residue":
                emb = hidden_states[idx].cpu().numpy()  # [seq_len, hidden_dim]
            elif embedding_type == "per-sequence":
                emb = hidden_states[idx].mean(dim=0).cpu().numpy()  # [hidden_dim]
            else:
                raise ValueError("embedding_type must be either 'per-sequence' or 'per-residue'")
            embeddings[protein_id] = emb

    print(f"Generated embeddings for {len(embeddings)} sequences.")
    return embeddings


def save_embeddings_to_hdf5(embeddings: Dict[str, torch.Tensor], h5_output_file: str):
    """
    Save embeddings to an HDF5 file.

    Parameters:
        embeddings (dict): {protein_id: embedding}
        h5_output_file (str): Path to the output HDF5 file.
    """
    print(f"Saving embeddings to HDF5 file: {h5_output_file}")
    try:
        with h5py.File(h5_output_file, "w") as h5f:
            for protein_id, emb in tqdm(embeddings.items(), desc="Saving Embeddings", total=len(embeddings)):
                sanitized_id = sanitize_hdf5_key(protein_id)
                # Handle potential duplicate keys
                if sanitized_id in h5f:
                    print(f"Warning: Dataset {sanitized_id} already exists. Overwriting.")
                    del h5f[sanitized_id]
                h5f.create_dataset(sanitized_id, data=emb)
        print(f"Embeddings successfully saved to {h5_output_file}.")
    except Exception as e:
        print(f"Error saving embeddings to HDF5: {e}")
        raise e


def inspect_hdf5_file(h5_output_file: str, max_keys: int = 10):
    """
    Inspect the HDF5 file by listing some keys and their dataset shapes.

    Parameters:
        h5_output_file (str): Path to the HDF5 file.
        max_keys (int): Number of keys to inspect.
    """
    print("\nInspecting the generated HDF5 file:")
    try:
        with h5py.File(h5_output_file, "r") as h5f:
            keys = list(h5f.keys())
            print(f"Total keys in HDF5 file: {len(keys)}")
            sample_keys = keys[:max_keys]
            for key in sample_keys:
                dataset = h5f[key]
                print(f"Protein ID: {key}, dataset shape: {dataset.shape}")
            if len(keys) > max_keys:
                print(f"... and {len(keys) - max_keys} more.")
    except Exception as e:
        print(f"Error inspecting HDF5 file: {e}")
        raise e


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generate protein embeddings using ProtT5.")
    parser.add_argument(
        "--fasta",
        type=str,
        required=True,
        help="Path to the input FASTA file containing protein sequences."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output HDF5 file to save embeddings."
    )
    parser.add_argument(
        "--embedding_type",
        type=str,
        choices=["per-sequence", "per-residue"],
        default="per-sequence",
        help="Type of embeddings to generate: 'per-sequence' (default) or 'per-residue'."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of sequences to process in each batch. Default is 8."
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU for embedding generation if available."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Rostlab/prot_t5_xl_uniref50",
        help="Name of the pretrained ProtT5 model to use."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum token length for the tokenizer. Default is 1024."
    )
    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Load protein sequences from FASTA file
    sequences = load_fasta(args.fasta)

    if not sequences:
        print("No valid sequences found. Exiting.")
        return

    # Load ProtT5 model and tokenizer
    tokenizer, model, device = load_prot_t5_model(model_name=args.model_name, use_gpu=args.use_gpu)

    # Generate embeddings
    embeddings = generate_embeddings_in_batches(
        sequences=sequences,
        tokenizer=tokenizer,
        model=model,
        device=device,
        embedding_type=args.embedding_type,
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    # Save embeddings to HDF5
    save_embeddings_to_hdf5(embeddings, args.output)

    # Inspect the HDF5 file
    inspect_hdf5_file(args.output)


if __name__ == "__main__":
    main()
