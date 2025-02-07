import logging
import os
import numpy as np
import pandas as pd

from config.config import (
    TRAINING_EMBEDDINGS_PATH_FOLDS,
    TEST_EMBEDDINGS_PATH,
    TRAINING_LABELS_PATH,
    TEST_LABELS_PATH,
    ALL_LABELS,
    PLOT_SAVE_DIR,
    N_COMPONENTS
)
from data_processing.data_loader import (
    load_embeddings,
    load_labels,
    merge_embeddings_labels
)
from data_processing.data_preprocessing_target import preprocess_data
from models.training import (
    train_random_forest,
    train_logistic_regression,
    train_svm,
    train_k_neighbors
)
# -- Remove combine_mccs since metrics.py no longer has it
from evaluation.metrics import (
    calculate_evaluation_metrics,
    generate_table,
    save_metrics_to_csv,
    save_table_to_file
)
from evaluation.confusion_matrix_plot import compute_confusion_matrix
from evaluation.curves import (
    compute_learning_curve,
    compute_roc_curves,
    compute_precision_recall_curve
)
from visualization.plots import (
    plot_comparative_bar_graph
)

from blast.blast_predictor import (
    load_blast_hits,
    load_blast_labels,
    run_blast_predictor,
    compute_confusion_matrix_blast
)

def main():
    # Configure logging if not already configured
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("pipeline.log", mode='a')
            ]
        )
    
    # Ensure plot save directory exists
    if not os.path.exists(PLOT_SAVE_DIR):
        os.makedirs(PLOT_SAVE_DIR)
        logging.info(f"Created plot save directory at {PLOT_SAVE_DIR}")

    logging.info("Loading Training Embeddings...")
    training_embeddings_df = load_embeddings(TRAINING_EMBEDDINGS_PATH_FOLDS)
    if training_embeddings_df is None:
        logging.error("Failed to load training embeddings.")
        return

    logging.info("Loading Training Labels...")
    training_labels_df = load_labels(TRAINING_LABELS_PATH)
    if training_labels_df is None:
        logging.error("Failed to load training labels.")
        return

    logging.info("Merging Training Embeddings and Labels...")
    merged_train_df = merge_embeddings_labels(training_embeddings_df, training_labels_df)
    if merged_train_df is None or merged_train_df.empty:
        logging.error("Failed to merge or no data after merging.")
        return

    logging.info("Preprocessing Training Data...")
    try:
        X_train, y_train_encoded, label_encoder = preprocess_data(merged_train_df)
    except ValueError as ve:
        logging.error(f"Preprocessing failed: {ve}")
        return

    logging.info("Loading Test Embeddings...")
    test_embeddings_df = load_embeddings(TEST_EMBEDDINGS_PATH)
    if test_embeddings_df is None:
        logging.error("Failed to load test embeddings.")
        return

    logging.info("Loading Test Labels...")
    test_labels_df = load_labels(TEST_LABELS_PATH)
    if test_labels_df is None:
        logging.error("Failed to load test labels.")
        return

    logging.info("Merging Test Embeddings and Labels...")
    merged_test_df = merge_embeddings_labels(test_embeddings_df, test_labels_df)
    if merged_test_df is None or merged_test_df.empty:
        logging.error("Failed to merge or no data after merging.")
        return

    logging.info("Preprocessing Test Data...")
    try:
        X_test, y_test_encoded, _ = preprocess_data(merged_test_df)
    except ValueError as ve:
        logging.error(f"Preprocessing failed: {ve}")
        return

    # Check class distribution
    unique_classes, counts = np.unique(y_train_encoded, return_counts=True)
    logging.info("Training Class Distribution Before Training:")
    for cls, cnt in zip(label_encoder.inverse_transform(unique_classes), counts):
        logging.info(f"{cls}: {cnt}")
    if len(unique_classes) < 2:
        logging.error("Training data contains only one class. Training aborted.")
        return

    # -------------------- BLAST Predictor Integration --------------------
    logging.info("Starting BLAST predictor integration...")
    logging.info("Loading BLAST Hits...")
    blast_hits_df = load_blast_hits(blast_results_path='./data/blast_results.tsv')
    if blast_hits_df.empty:
        logging.error("BLAST hits data is empty or failed to load.")
    else:
        logging.info("Loading BLAST Labels...")
        type_map, target_map = load_blast_labels(labels_file_path='./data/ToxinTypes_labelTarget_3.csv')
        if not type_map or not target_map:
            logging.error("Failed to load BLAST label mappings.")
        else:
            logging.info("Running BLAST Predictor...")
            blast_predictions_df = run_blast_predictor(blast_hits_df, type_map, target_map)
            logging.info(f"Number of BLAST predictions: {blast_predictions_df.shape[0]}")
    
            # Replace 'sseqid' with your actual identifier column in merged_test_df that matches 'qseqid'
            # For example, if 'sseqid' in BLAST corresponds to 'sequence_id' in test data:
            merged_test_with_blast = pd.merge(
                merged_test_df,
                blast_predictions_df[['qseqid', 'predicted_exotoxin_type', 'predicted_target']],
                left_on='ID',  # Replace with your actual column name in merged_test_df
                right_on='qseqid',
                how='left'
            )

            print("merged_test_df columns:", merged_test_df.columns)
            print("blast_predictions_df columns:", blast_predictions_df.columns)

    
            # Add logging to verify the merge
            logging.info(f"Merged Test with BLAST (first 5 rows):\n{merged_test_with_blast.head()}")
            logging.info(f"Number of BLAST predictions merged: {merged_test_with_blast['predicted_exotoxin_type'].notna().sum()}")
    
            # Handle missing BLAST predictions
            merged_test_with_blast['predicted_exotoxin_type'].fillna('Unknown', inplace=True)
            merged_test_with_blast['predicted_target'].fillna('Unknown', inplace=True)
    
            replacement_map = {
                "Non-vertebrate": "Non-Vertebrate",
                "vertebrate": "Vertebrate",
                "non-vertebrate": "Non-Vertebrate",  # In case of lowercase variations
                # Add any other necessary mappings
            }
            merged_test_with_blast['predicted_target'] = merged_test_with_blast['predicted_target'].replace(replacement_map)

            # Encode BLAST predictions
            from sklearn.preprocessing import LabelEncoder
    
            # Option 1: Use the same label_encoder if possible
            # Check if 'Unknown' is in label_encoder classes
            if 'Unknown' not in label_encoder.classes_:
                label_encoder.classes_ = np.append(label_encoder.classes_, 'Unknown')
                logging.info("Added 'Unknown' to label_encoder classes.")
    
            try:
                merged_test_with_blast['predicted_target_encoded'] = label_encoder.transform(
                merged_test_with_blast['predicted_target']
            )
                logging.info("BLAST predictions successfully encoded using the existing label_encoder.")
            except ValueError as ve:
                logging.error(f"Error encoding BLAST predictions: {ve}")
                return
    
            # Extract BLAST predictions
            y_pred_blast = merged_test_with_blast['predicted_target_encoded'].values
            logging.info(f"y_pred_blast length: {len(y_pred_blast)}")
    
    # ----------------------------------------------------------------------

    # Train models
    logging.info("Training Random Forest Classifier...")
    rf_model, rf_grid_search, rf_param_grid = train_random_forest(X_train, y_train_encoded, N_COMPONENTS)

    logging.info("Training Logistic Regression Classifier...")
    lr_model, lr_grid_search, lr_param_grid = train_logistic_regression(X_train, y_train_encoded, N_COMPONENTS)

    logging.info("Training Support Vector Machine Classifier...")
    svm_model, svm_grid_search, svm_param_grid = train_svm(X_train, y_train_encoded, N_COMPONENTS)

    logging.info("Training K-Nearest Neighbors Classifier...")
    knn_model, knn_grid_search, knn_param_grid = train_k_neighbors(X_train, y_train_encoded, N_COMPONENTS)



    # Predictions
    test_features = X_test
    y_pred_rf  = rf_model.predict(test_features)
    y_pred_lr  = lr_model.predict(test_features)
    y_pred_svm = svm_model.predict(test_features)
    y_pred_knn = knn_model.predict(test_features)

    # Add BLAST predictions if available
    if not blast_hits_df.empty and 'y_pred_blast' in locals():
        # Ensure that y_pred_blast aligns with y_test_encoded
        if len(y_pred_blast) != len(y_test_encoded):
            logging.warning("y_pred_blast length does not match y_test_encoded length. Adjusting accordingly.")
            # Adjust y_pred_blast to match y_test_encoded if possible
            # This depends on how the merge was performed
            # For simplicity, assuming they align
        # Else, proceed as is
        pass
    else:
        y_pred_blast = None

    # ---------------------------------------------------------------------
    # Evaluate each classifier & produce an Error-Bar Table (MCC, Accuracy, etc.)
    # ---------------------------------------------------------------------
    # Evaluation Metrics with Error Bars
    logging.info("Calculating evaluation metrics...")
    metrics_dict = {
        "RandomForest": calculate_evaluation_metrics(y_test_encoded, y_pred_rf, label_encoder=label_encoder),
        "LogisticRegression": calculate_evaluation_metrics(y_test_encoded, y_pred_lr, label_encoder=label_encoder),
        "SVM": calculate_evaluation_metrics(y_test_encoded, y_pred_svm, label_encoder=label_encoder),
        "KNN": calculate_evaluation_metrics(y_test_encoded, y_pred_knn, label_encoder=label_encoder)
    }


    # Add BLAST metrics if available
    if y_pred_blast is not None:
        metrics_dict["BLAST"] = calculate_evaluation_metrics(y_test_encoded, y_pred_blast, label_encoder=label_encoder)

    # Log the metrics_dict keys
    logging.info(f"Metrics calculated for: {list(metrics_dict.keys())}")

    # Generate and Save Tables
    for model_name, metrics in metrics_dict.items():
        table = generate_table(metrics)
        print(f"\n{model_name} Metrics Table:\n{table}")
        save_metrics_to_csv(metrics, file_name=f"{model_name}_metrics.csv")
        save_table_to_file(table, file_name=f"{model_name}_metrics_table.txt")

    # **New Section: Aggregate MCC and SE into combined_mcc_df**
    logging.info("Aggregating MCC and SE for all classifiers...")
    data = {
        "Predictor": [],
        "MCC": [],
        "MCC_SE": []
    }

    for model_name, metrics in metrics_dict.items():
        mcc, mcc_se = metrics.get('MCC', (0.0, 0.0))
        data["Predictor"].append(model_name)
        data["MCC"].append(mcc)
        data["MCC_SE"].append(mcc_se)

    combined_mcc_df = pd.DataFrame(data)
    logging.info(f"Combined MCC DataFrame:\n{combined_mcc_df}")

    # Check if BLAST is included
    if "BLAST" not in combined_mcc_df["Predictor"].values:
        logging.error("BLAST is not included in the Combined MCC DataFrame.")
    else:
        logging.info("BLAST is successfully included in the Combined MCC DataFrame.")

    bar_colors = ["#43A047", "#8E24AA"] 
    # **Plotting the Comparative Bar Graph with Error Bars**
    logging.info("Plotting comparative bar graph with error bars...")
    plot_comparative_bar_graph(
        combined_mcc_df, 
        bar_colors=bar_colors,
        title="MCC Comparison of Predictors for Exotoxin Target Classification Using Method 1",
        save_path=os.path.join(PLOT_SAVE_DIR, "comparison_graph_with_error_bars.png")
    )
    logging.info("Comparative bar graph plotted successfully.")

    # Confusion Matrices
    logging.info("Computing confusion matrices...")
    compute_confusion_matrix(rf_model,  test_features, y_test_encoded, ALL_LABELS, "test", "flat", "RandomForest", label_encoder)
    compute_confusion_matrix(lr_model,  test_features, y_test_encoded, ALL_LABELS, "test", "flat", "LogisticRegression", label_encoder)
    compute_confusion_matrix(svm_model, test_features, y_test_encoded, ALL_LABELS, "test", "flat", "SVM", label_encoder)
    compute_confusion_matrix(knn_model, test_features, y_test_encoded, ALL_LABELS, "test", "flat", "KNN", label_encoder)


    # Learning Curves
    logging.info("Computing learning curves...")
    compute_learning_curve(rf_model,  X_train, y_train_encoded, "train", "flat", "RandomForest")
    compute_learning_curve(lr_model,  X_train, y_train_encoded, "train", "flat", "LogisticRegression")
    compute_learning_curve(svm_model, X_train, y_train_encoded, "train", "flat", "SVM")
    compute_learning_curve(knn_model, X_train, y_train_encoded, "train", "flat", "KNN")

    if y_pred_blast is not None:
            # Learning curves for BLAST are not applicable as it's not a trainable model
            pass

    # ROC and Precision-Recall Curves
    logging.info("Computing ROC and Precision-Recall curves...")
    models_for_curves      = [rf_model, lr_model, svm_model, knn_model]
    model_names_for_curves = ["RandomForest", "LogisticRegression", "SVM", "KNN"]

    # BLAST does not provide probability estimates, so ROC curves might not be applicable
    compute_roc_curves(
        models_for_curves,
        model_names_for_curves,
        test_features,
        y_test_encoded,
        ALL_LABELS,
        "test",
        "flat"
    )
    compute_precision_recall_curve(
        models_for_curves,
        model_names_for_curves,
        test_features,
        y_test_encoded,
        ALL_LABELS,
        "test",
        "flat"
    )

if __name__ == "__main__":
    main()
