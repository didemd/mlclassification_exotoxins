#  Bacterial Exotoxin Classification

Main Code directory for the Master Thesis Project

Relevant modules, files and directories can be found in the  subdirectories.

Directory Name: Code/python
Creation Date: 2023-03-15
Author: Didem Dost

# Script description

Script Descriptions:
Target Classification (Binary Predictor)

    main_target_fold_standard_split.py

        Input: Fold-based embeddings.

        Predictor Type: Binary classification (Target-based).

        Splitting Approach: Standard random data splitting (80/20 split).

    main_target_redundancy_reduction_split.py

        Input: Sequence-based embeddings.

        Predictor Type: Binary classification (Target-based).

        Splitting Approach: Redundancy reduction-based splitting.

    main_target_standard_split.py

        Input: Sequence-based embeddings.

        Predictor Type: Binary classification (Target-based).

        Splitting Approach: Standard random data splitting (80/20 split).

Type Classification (Multiclass Predictor)

    main_type_fold_standard_split.py

        Input: Fold-based embeddings (cross-validation folds).

        Predictor Type: Multiclass classification (Type-based).

        Splitting Approach: Standard random data splitting (80/20 split).

    main_type_redundancy_reduction_split.py

        Input: Sequence-based embeddings.

        Predictor Type: Multiclass classification (Type-based).

        Splitting Approach: Redundancy reduction-based splitting.

    main_type_standard_split.py

        Input: Sequence-based embeddings.

        Predictor Type: Multiclass classification (Type-based).

        Splitting Approach: Standard random data splitting (80/20 split).


## Objective

This project implements machine learning models to classify bacterial exotoxins based on their types and targets. The codebase provides tools for data processing, model training, evaluation, and visualization of results.

## Structure

- **blast/**: BLAST-based prediction functionality
- **config/**: Configuration settings and constants
- **data_processing/**: Data loading, preprocessing, and transformation
- **evaluation/**: Metrics calculation and model evaluation tools
- **visualization/**: Plotting and visualization utilities
- **main_*.py**: Main execution scripts for different classification tasks
