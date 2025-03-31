# mlclassification_exotoxins
Master Thesis Project on Machine Learning Classification Based on Types and Targets

In my master thesis, I explored advanced machine learning methods to classify bacterial exotoxins according to two critical predictors:

    Exotoxin Type: Identifying the category of exotoxins (e.g., Type I, II, III, IV, or Unknown types), using sequence-based features, embeddings, and computationally extracted descriptors.

    Exotoxin Target: Classifying exotoxins based on their biological targets, vertebrates and non-vertebrates, which is vital for therapeutic and diagnostic purposes.

I have developed Python scripts utilizing machine learning algorithms like Random Forest, Logistic Regression, SVM, KNN, and hierarchical approaches to investigate their predictive performances. The scripts process input data, create feature embeddings, train machine learning models, evaluate their performance using standard metrics (accuracy, MCC, ROC curves, precision-recall curves), and analyze the results comprehensively.

# Scripts available in the pipeline

Data Splitting Methods Used:

Method 1: Redundancy Reduction Split

- After redundancy reduction with a sequence identity threshold of 30%, sequences exhibiting more than 30% similarity to any other sequence are excluded from the training set and instead assigned to the test set. This ensures the training set contains unique, non-redundant sequences, minimizing biases due to redundant sequences and enhancing model generalizability, while still utilizing the entire dataset for evaluation.

Method 2: Standard Split

- It involves performing a stratified random split on the non-redundant dataset. Using Scikit-Learn’s train_test_split function (version 1.6.1) with a fixed random state of 42 for reproducibility, the dataset is partitioned into 80% training and 20% test sets.

### 1. Setup
First, give the script execution permissions:
```bash
chmod +x run.sh
```

### 2. Run the Script
```bash
./run.sh
```

You will be prompted to select a script to execute:

    - Choose a specific script (main_target.py, main_target_folds_split.py, main_target_split.py, main_type_folds_split.py, main_type_split.py, main_type.py) 
    - Select "All" to run all main scripts sequentially.

### 3. Virtual Environmnet

The script automatically:

    - Creates and activates a virtual environment
    - Installs dependencies from requirements.txt
    - Runs the selected script(s) from Code/python/
    - Deactivates the virtual environment when done
