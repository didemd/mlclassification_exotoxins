# mlclassification_exotoxins
Master Thesis Project on Machine Learning Classification Based on Types and Targets


## Running the Code

To execute the main scripts located in `Code/python/`, use the provided `run.sh` bash script.

### 1. Setup
First, give the script execution permissions:
```bash
chmod +x run.sh
```

### 2. Run the Script
```bash
./run.sh
```

You will be prompted to select a script to execute (Right now main_type.py and main_target.py working):

    - Choose a specific script (e.g., main_target.py) or
    - Select "All" to run all main scripts sequentially.

### 3. Virtual Environmnet

The script automatically:

    Creates and activates a virtual environment
    Installs dependencies from requirements.txt
    Runs the selected script(s) from Code/python/
    Deactivates the virtual environment when done
