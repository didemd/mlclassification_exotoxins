#!/bin/bash

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Python scripts
python Code/python/main_type.py


# Deactivate virtual environment
deactivate
