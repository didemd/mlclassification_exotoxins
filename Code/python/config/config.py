"""
A config.py file allows you to keep all your constants, paths, and 
general configuration in one place.
"""
import os
import logging
import warnings
import numpy as np

# --------------------- Configure Logging and Warnings ---------------------
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set random seed for reproducibility
np.random.seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define directory for saving plots
#PLOT_SAVE_DIR = os.getenv('PLOT_SAVE_DIR', os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "Figures")))
#PLOT_SAVE_DIR_TYPE = os.getenv('PLOT_SAVE_DIR', os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "Figures_Type")))

PLOT_SAVE_DIR = os.getenv('PLOT_SAVE_DIR', "Figures")
PLOT_SAVE_DIR_TYPE = os.getenv('PLOT_SAVE_DIR', "Figures")

os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
os.makedirs(PLOT_SAVE_DIR_TYPE, exist_ok=True)

# Paths (adjust as needed)
TRAINING_EMBEDDINGS_PATH = os.getenv('TRAINING_EMBEDDINGS_PATH', "./Data/derived/per_residue_embeddings_training.h5")
TEST_EMBEDDINGS_PATH     = os.getenv('TEST_EMBEDDINGS_PATH', "./Data/derived/per_residue_embeddings_test.h5")
TRAINING_LABELS_PATH     = os.getenv('TRAINING_LABELS_PATH', "./Data/raw/ToxinTypes_labelTarget_3.csv")
TEST_LABELS_PATH         = os.getenv('TEST_LABELS_PATH', "./Data/raw/ToxinTypes_labelTarget_3.csv")
#TRAINING_EMBEDDINGS_PATH_FOLDS = os.getenv('TRAINING_EMBEDDINGS_PATH_FOLDS', "./data/folds_with_labels_updated.h5")
TRAINING_EMBEDDINGS_PATH_FOLDS = os.getenv('TRAINING_EMBEDDINGS_PATH_FOLDS', "./Data/derived/folds_with_labels.h5")



BLAST_RESULTS_PATH = os.getenv('BLAST_RESULTS_PATH', "./Data/derived/blast_results.tsv")
MODEL_SAVE_DIR = os.getenv('MODEL_SAVE_DIR', "Predictors")
PREDICTOR_PATH = os.getenv('PREDICTOR_PATH', "Predictors")

# Define directory for saving statistics and metrics
STATS_DIR = os.getenv('STATS_DIR', "Stats")
os.makedirs(STATS_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


# Label order for confusion matrix and other plots
ALL_LABELS = ['Non-Vertebrate', 'Vertebrate']

# Number of PCA components
N_COMPONENTS = int(os.getenv('N_COMPONENTS', 50))

# Validate paths
for path in [TRAINING_EMBEDDINGS_PATH, TEST_EMBEDDINGS_PATH, TRAINING_LABELS_PATH, TEST_LABELS_PATH]:
    if not os.path.exists(path):
        logging.warning(f"Path does not exist: {path}")
