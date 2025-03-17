#!/bin/bash

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Define script directory
SCRIPT_DIR="Code/python"

# Allow user to select a script to run
echo "Select a script to run:"
select script in \
    main_target_redundancy_reduction_split.py \
    main_target_standard_split.py \
    main_target_fold_standard_split.py \
    main_type_redundancy_reduction_split.py \
    main_type_standard_split.py \
    main_type_fold_standard_split.py \
    "All"; do
    case $script in
        "All")
            echo "Running all main scripts..."
            for file in "$SCRIPT_DIR"/main_*.py; do
                echo "Executing $file..."
                python "$file"
            done
            break
            ;;
        "")
            echo "Invalid choice. Please try again."
            ;;
        *)
            echo "Running $script..."
            python "$SCRIPT_DIR/$script"
            break
            ;;
    esac
done

# Deactivate virtual environment
deactivate
