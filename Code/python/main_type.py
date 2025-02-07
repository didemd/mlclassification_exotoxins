# my_project/main_type.py

import logging
import os
import numpy as np
import pandas as pd

from config.config import (
    TRAINING_EMBEDDINGS_PATH,
    TEST_EMBEDDINGS_PATH,
    TRAINING_LABELS_PATH,
    TEST_LABELS_PATH,
    ALL_LABELS,
    PLOT_SAVE_DIR,
    N_COMPONENTS,
    BLAST_RESULTS_PATH,
    STATS_DIR,
    MODEL_SAVE_DIR,
    PREDICTOR_PATH
)
from data_processing.data_loader import load_embeddings, load_labels, merge_embeddings_labels
from data_processing.data_preprocessing_type import preprocess_data
from models.training import (
    train_random_forest,
    train_logistic_regression,
    train_svm,
    train_k_neighbors
)
from models.hierarchical import (
    train_hierarchical_classifier_rf,
    train_hierarchical_classifier_svm
)

from blast.blast_predictor import (
    load_blast_hits,
    load_blast_labels,
    run_blast_predictor,
    extract_features_and_labels,
    preprocess_features,
    train_model,
    evaluate_model
)

from sklearn.model_selection import train_test_split
import os
# -- Import only what's needed from metrics.py
from evaluation.metrics import (
    calculate_evaluation_metrics,
    generate_table,
    save_metrics_to_csv,
    save_table_to_file
)

from visualization.plots import(
    plot_2x2_learning_curves,
    plot_2x2_roc_curves,
    plot_2x2_precision_recall_curves,
    plot_comparative_bar_graph,
    compute_learning_curve_data,
    compute_confusion_matrix,
    plot_2x2_confusion_matrices,
    generate_class_metrics_table
)

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main_type():
    """
    Main pipeline for 'Type' prediction, including hierarchical classifiers, 
    BLAST integration, confusion matrices, MCC comparisons, etc.
    """
    # ----------------------------------------------------------------------
    # 1. Define label order for confusion matrices, etc.
    # ----------------------------------------------------------------------
    all_labels = ['Type_I', 'Type_II', 'Type_III', 'Type_IV', 'Unknown']

    # ----------------------------------------------------------------------
    # 2. Define input file paths (or read from config.py)
    # ----------------------------------------------------------------------
    training_embeddings_path = TRAINING_EMBEDDINGS_PATH
    test_embeddings_path     = TEST_EMBEDDINGS_PATH
    training_labels_path     = TRAINING_LABELS_PATH
    test_labels_path         = TEST_LABELS_PATH
    blast_results_path       = BLAST_RESULTS_PATH
    model_save_dir           = MODEL_SAVE_DIR

    # ----------------------------------------------------------------------
    # 3. Load and merge training data
    # ----------------------------------------------------------------------
    logging.info("Loading Training Embeddings...")
    training_embeddings_df = load_embeddings(training_embeddings_path)
    if training_embeddings_df is None:
        logging.error("Failed to load training embeddings.")
        return

    logging.info("Loading Training Labels...")
    training_labels_df = load_labels(training_labels_path, selected_column='type')
    if training_labels_df is None:
        logging.error("Failed to load training labels.")
        return

    logging.info("Merging Training Embeddings and Labels...")
    merged_train_df = merge_embeddings_labels(training_embeddings_df, training_labels_df, selected_column='type')
    if merged_train_df is None or merged_train_df.empty:
        logging.error("Failed to merge training embeddings and labels.")
        return

    # ----------------------------------------------------------------------
    # 4. Preprocess training data
    # ----------------------------------------------------------------------
    logging.info("Preprocessing Training Data...")
    X_train_main, y_train_main, label_encoder_main = preprocess_data(merged_train_df)

    # If hierarchical training needs the full DataFrame:
    feature_columns_main = [f'feature_{i}' for i in range(X_train_main.shape[1])]
    training_df_main = pd.DataFrame(X_train_main, columns=feature_columns_main)
    training_df_main['type'] = label_encoder_main.inverse_transform(y_train_main)

    # ----------------------------------------------------------------------
    # 5. Load and merge test data
    # ----------------------------------------------------------------------
    logging.info("Loading Test Embeddings...")
    test_embeddings_df = load_embeddings(test_embeddings_path)
    if test_embeddings_df is None:
        logging.error("Failed to load test embeddings.")
        return

    logging.info("Loading Test Labels...")
    test_labels_df = load_labels(test_labels_path, selected_column='type')
    if test_labels_df is None:
        logging.error("Failed to load test labels.")
        return

    logging.info("Merging Test Embeddings and Labels...")
    merged_test_df = merge_embeddings_labels(test_embeddings_df, test_labels_df, selected_column='type')
    if merged_test_df is None or merged_test_df.empty:
        logging.error("Failed to merge test embeddings and labels.")
        return

    logging.info("Preprocessing Test Data...")
    X_test_main, y_test_main, _ = preprocess_data(merged_test_df)

    test_df_main = pd.DataFrame(X_test_main, columns=feature_columns_main)
    test_df_main['type'] = label_encoder_main.inverse_transform(y_test_main)

    print("LabelEncoder classes:", label_encoder_main.classes_)

    # ----------------------------------------------------------------------
    # 6. Train hierarchical classifiers (RF & SVM)
    # ----------------------------------------------------------------------
    logging.info("Training Random Forest-based Hierarchical Classifier...")
    model_rf_hier = train_hierarchical_classifier_rf(training_df_main)

    logging.info("Training SVM-based Hierarchical Classifier...")
    model_svm_hier = train_hierarchical_classifier_svm(training_df_main)

    # ----------------------------------------------------------------------
    # 7. Train "flat" classifiers (RF, LR, SVM, KNN)
    # ----------------------------------------------------------------------
    n_components = 50

    logging.info("Training Random Forest Classifier...")
    rf_model, rf_grid_search, _ = train_random_forest(X_train_main, y_train_main, n_components)

    logging.info("Training Logistic Regression Classifier...")
    lr_model, lr_grid_search, lr_param_grid = train_logistic_regression(X_train_main, y_train_main, n_components)

    logging.info("Training Support Vector Machine Classifier...")
    svm_model, svm_grid_search, _ = train_svm(X_train_main, y_train_main, n_components)

    logging.info("Training K-Nearest Neighbors Classifier...")
    knn_model, knn_grid_search, _ = train_k_neighbors(X_train_main, y_train_main, n_components)

    # ----------------------------------------------------------------------
    # 8. Predictions
    # ----------------------------------------------------------------------
    test_features_main = X_test_main

    # After hierarchical predictions
    logging.info("Evaluating Hierarchical RF Classifier on Test Set...")
    y_pred_rf_hier = model_rf_hier.predict(test_features_main)
    try:
        y_pred_rf_hier_encoded = label_encoder_main.transform(y_pred_rf_hier)
    except ValueError as e:
        logging.error(f"Hierarchical RF Classifier encountered unseen labels: {e}")
        y_pred_rf_hier_encoded = np.full_like(y_test_main, fill_value=-1)  # Assign a default value or handle appropriately

    logging.info("Evaluating Hierarchical SVM Classifier on Test Set...")
    y_pred_svm_hier = model_svm_hier.predict(test_features_main)
    try:
        y_pred_svm_hier_encoded = label_encoder_main.transform(y_pred_svm_hier)
    except ValueError as e:
        logging.error(f"Hierarchical SVM Classifier encountered unseen labels: {e}")
        y_pred_svm_hier_encoded = np.full_like(y_test_main, fill_value=-1)  # Assign a default value or handle appropriately


    logging.info("Evaluating Random Forest Classifier on Test Set...")
    y_pred_rf = rf_model.predict(test_features_main)

    logging.info("Evaluating Logistic Regression Classifier on Test Set...")
    y_pred_lr = lr_model.predict(test_features_main)

    logging.info("Evaluating SVM Classifier on Test Set...")
    y_pred_svm = svm_model.predict(test_features_main)

    logging.info("Evaluating K-Nearest Neighbors Classifier on Test Set...")
    y_pred_knn = knn_model.predict(test_features_main)
     
    # ----------------------------------------------------------------------
    # 9. Evaluate classifiers with bootstrapped error bars
    # ----------------------------------------------------------------------
    # For each classifier, compute metrics, build table, print & save.

    # -- Hierarchical RF
    hier_rf_metrics = calculate_evaluation_metrics(
        y_test_main, 
        y_pred_rf_hier_encoded, 
        label_encoder=label_encoder_main
    )
    table_hier_rf = generate_table(hier_rf_metrics)
    print("\nHierarchical RF Metrics Table:\n", table_hier_rf)
    save_metrics_to_csv(hier_rf_metrics, file_name=os.path.join(STATS_DIR, "Hierarchical_RF_metrics.csv"))
    save_table_to_file(table_hier_rf, file_name=os.path.join(STATS_DIR, "Hierarchical_RF_metrics_table.txt"))

    hier_rf_class_metrics = generate_class_metrics_table(y_test_main, y_pred_rf_hier_encoded, all_labels)
    print("\n Hier RF (Flat) Per-Class Metrics Table:\n", hier_rf_class_metrics)
    hier_rf_class_metrics.to_csv(os.path.join(PLOT_SAVE_DIR, "Hier_RF_flat_per_class_metrics.csv"), index=False)

    # -- Hierarchical SVM
    hier_svm_metrics = calculate_evaluation_metrics(
        y_test_main, 
        y_pred_svm_hier_encoded, 
        label_encoder=label_encoder_main
    )    
    table_hier_svm = generate_table(hier_svm_metrics)
    print("\nHierarchical SVM Metrics Table:\n", table_hier_svm)
    save_metrics_to_csv(hier_svm_metrics, file_name=os.path.join(STATS_DIR, "Hierarchical_SVM_metrics.csv"))
    save_table_to_file(table_hier_svm, file_name=os.path.join(STATS_DIR, "Hierarchical_SVM_metrics_table.txt"))

    hier_svm_class_metrics = generate_class_metrics_table(y_test_main, y_pred_svm_hier_encoded, all_labels)
    print("\n Hier SVM (Flat) Per-Class Metrics Table:\n", hier_svm_class_metrics)
    hier_svm_class_metrics.to_csv(os.path.join(PLOT_SAVE_DIR, "Hier_SVM_flat_per_class_metrics.csv"), index=False)

    # Logging unique predictions for debugging
    unique_pred_rf_hier = np.unique(y_pred_rf_hier)
    unique_pred_svm_hier = np.unique(y_pred_svm_hier)
    logging.info(f"Unique predictions from Hierarchical RF: {unique_pred_rf_hier}")
    logging.info(f"Unique predictions from Hierarchical SVM: {unique_pred_svm_hier}")

    # -- Random Forest (flat)
    rf_metrics = calculate_evaluation_metrics(y_test_main, y_pred_rf, label_encoder=label_encoder_main)
    table_rf = generate_table(rf_metrics)
    print("\nRandom Forest (flat) Metrics Table:\n", table_rf)
    save_metrics_to_csv(rf_metrics, file_name=os.path.join(STATS_DIR, "RandomForest_flat_metrics.csv"))
    save_table_to_file(table_rf, file_name=os.path.join(STATS_DIR, "RandomForest_flat_metrics_table.txt"))

    rf_class_metrics = generate_class_metrics_table(y_test_main, y_pred_rf, all_labels)
    print("\nRandom Forest (Flat) Per-Class Metrics Table:\n", rf_class_metrics)
    rf_class_metrics.to_csv(os.path.join(PLOT_SAVE_DIR, "RandomForest_flat_per_class_metrics.csv"), index=False)

    # -- Logistic Regression
    lr_metrics = calculate_evaluation_metrics(y_test_main, y_pred_lr, label_encoder=label_encoder_main)
    table_lr = generate_table(lr_metrics)
    print("\nLogistic Regression Metrics Table:\n", table_lr)
    save_metrics_to_csv(lr_metrics, file_name=os.path.join(STATS_DIR, "LogisticRegression_metrics.csv"))
    save_table_to_file(table_lr, file_name=os.path.join(STATS_DIR, "LogisticRegression_metrics_table.txt"))

    lr_class_metrics = generate_class_metrics_table(y_test_main, y_pred_lr, all_labels)
    print("\nLogistic Regression (Flat) Per-Class Metrics Table:\n", lr_class_metrics)
    lr_class_metrics.to_csv(os.path.join(PLOT_SAVE_DIR, "LogisticRegression_flat_per_class_metrics.csv"), index=False)

    # -- SVM
    svm_metrics = calculate_evaluation_metrics(y_test_main, y_pred_svm, label_encoder=label_encoder_main)
    table_svm = generate_table(svm_metrics)
    print("\nSVM Metrics Table:\n", table_svm)
    save_metrics_to_csv(svm_metrics, file_name=os.path.join(STATS_DIR, "SVM_metrics.csv"))
    save_table_to_file(table_svm, file_name=os.path.join(STATS_DIR, "SVM_metrics_table.txt"))

    svm_class_metrics = generate_class_metrics_table(y_test_main, y_pred_svm, all_labels)
    print("\n SVM (Flat) Per-Class Metrics Table:\n", svm_class_metrics)
    svm_class_metrics.to_csv(os.path.join(PLOT_SAVE_DIR, "SVM_flat_per_class_metrics.csv"), index=False)

    # -- KNN
    knn_metrics = calculate_evaluation_metrics(y_test_main, y_pred_knn, label_encoder=label_encoder_main)
    table_knn = generate_table(knn_metrics)
    print("\nKNN Metrics Table:\n", table_knn)
    save_metrics_to_csv(knn_metrics, file_name=os.path.join(STATS_DIR, "KNN_metrics.csv"))
    save_table_to_file(table_knn, file_name=os.path.join(STATS_DIR, "KNN_metrics_table.txt"))

    knn_class_metrics = generate_class_metrics_table(y_test_main, y_pred_knn, all_labels)
    print("\n SVM (Flat) Per-Class Metrics Table:\n", knn_class_metrics)
    knn_class_metrics.to_csv(os.path.join(PLOT_SAVE_DIR, "KNN_flat_per_class_metrics.csv"), index=False)
    # ----------------------------------------------------------------------
    # 10. BLAST Integration & Evaluation
    # ----------------------------------------------------------------------
    # Following the assistant's guidance, ensure distinct variable names

    # Step 1: Load BLAST hits
    blast_hits = load_blast_hits(blast_results_path)
    if blast_hits.empty:
        logging.error("No BLAST hits loaded. Exiting.")
        return

    # Step 2: Load label mappings
    type_map, target_map = load_blast_labels(training_labels_path)
    if not type_map and not target_map:
        logging.error("Label mappings not loaded. Exiting.")
        return

    # Step 3: Run predictor to get top hits with labels
    predicted_hits = run_blast_predictor(blast_hits, type_map, target_map)
    if predicted_hits.empty:
        logging.error("No predicted hits to process. Exiting.")
        return

    # Optional: Verify the columns in predicted_hits
    print("Available columns in predicted_hits:", predicted_hits.columns.tolist())

    # Step 4: Feature Extraction for Exotoxin Type Prediction
    X_blast, y_blast = extract_features_and_labels(predicted_hits, label_type='predicted_exotoxin_type')
    if X_blast is None or y_blast is None:
        logging.error("Failed to extract features and labels for exotoxin type. Exiting.")
        return

    # Step 5: Preprocess Features
    X_blast_scaled = preprocess_features(X_blast)

    # Step 6: Split into Training and Testing Sets (Renamed)
    X_train_blast, X_test_blast, y_train_blast, y_test_blast = train_test_split(
        X_blast_scaled, y_blast, test_size=0.2, random_state=42, stratify=y_blast
    )
    logging.info(f"BLAST Training set size: {X_train_blast.shape[0]} samples")
    logging.info(f"BLAST Testing set size: {X_test_blast.shape[0]} samples")

    # Initialize LabelEncoder for BLAST (Separate from Main)
    blast_label_encoder = LabelEncoder()
    blast_label_encoder.fit(y_train_blast)

    # Step 7: Train the BLAST Model (Separate)
    model_path_blast = os.path.join(model_save_dir, 'random_forest_blast.pkl')
    model_blast = train_model(X_train_blast, y_train_blast, model_path=model_path_blast)

    # Step 8: Evaluate the BLAST Model
    evaluate_model(model_blast, X_test_blast, y_test_blast)

    # Evaluate BLAST Metrics (Separate)
    logging.info("Calculating Evaluation Metrics for BLAST Predictor...")
    try:
        y_pred_blast = model_blast.predict(X_test_blast)  # Corrected variable name
        y_pred_blast_encoded = blast_label_encoder.transform(y_pred_blast)
    except ValueError as e:
        logging.error(f"BLAST Predictor encountered unseen labels: {e}")
        y_pred_blast_encoded = np.full_like(y_test_blast, fill_value=-1)  # Assign a default value or handle appropriately

    blast_metrics = calculate_evaluation_metrics(
        y_test_blast,  # True labels
        y_pred_blast,  # Predicted labels
        label_encoder=blast_label_encoder
    )

    table_blast = generate_table(blast_metrics)
    print("\nBLAST Exotoxin Type Metrics Table:\n", table_blast)
    save_metrics_to_csv(blast_metrics, os.path.join(STATS_DIR, "BLAST_Exotoxin_metrics.csv"))
    save_table_to_file(table_blast, os.path.join(STATS_DIR, "BLAST_Exotoxin_metrics_table.txt"))

    # ----------------------------------------------------------------------
    # 11. Confusion Matrices
    # ----------------------------------------------------------------------
    logging.info("Computing confusion matrices...")

    # Assuming compute_confusion_matrix is updated to handle separate datasets
    # If not, ensure to pass the correct dataset variables

    # ----------------------------------------------------------------------
    # 12. Aggregate MCC and SE into combined_mcc_df
    # ----------------------------------------------------------------------
    logging.info("Aggregating MCC and SE for all classifiers...")
    data = {
        "Predictor": [],
        "MCC": [],
        "MCC_SE": []
    }

    all_predictors = ["RF", "LR", "SVM", "KNN", 
                      "Hier_RF", "Hier_SVM", "BLAST"]

    metrics_dict = {
        "RF": rf_metrics,
        "LR": lr_metrics,
        "SVM": svm_metrics,
        "KNN": knn_metrics,
        "Hier_RF": hier_rf_metrics,
        "Hier_SVM": hier_svm_metrics,
        "BLAST": blast_metrics
    }

    for model_name in all_predictors:
        metrics = metrics_dict.get(model_name, {})
        mcc_tuple = metrics.get('MCC', (0.0, 0.0))  # Ensure it's a tuple (MCC, SE)
        if isinstance(mcc_tuple, tuple) and len(mcc_tuple) == 2:
            mcc, mcc_se = mcc_tuple
        else:
            mcc, mcc_se = (0.0, 0.0)
        data["Predictor"].append(model_name)
        data["MCC"].append(mcc)
        data["MCC_SE"].append(mcc_se)

    combined_mcc_df = pd.DataFrame(data)
    logging.info(f"Combined MCC DataFrame:\n{combined_mcc_df}")

    bar_colors = [      # Color-blind friendly colors
        "#0072B2",  
        "#E69F00",  
        "#009E73",  
        "#CC79A7", 
        "#F0E442",  
        "#56B4E9",  
        "#A6761D"   
    ]

    logging.info("Plotting comparative bar graph with error bars...")
    plot_comparative_bar_graph(
        combined_mcc_df, 
        bar_colors=bar_colors,
        save_path=os.path.join(PLOT_SAVE_DIR, "comparison_graph_with_error_bars.png")
    )

    # ----------------------------------------------------------------------
    # 13. Learning Curves (for flat models)
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # 14. ROC and Precision-Recall curves (Multiclass)
    # ----------------------------------------------------------------------
    logging.info("Computing ROC and Precision-Recall curves...")

    models_for_curves      = [rf_model, lr_model, svm_model, knn_model]
    model_names_for_curves = ["RandomForest", "LogisticRegression", "SVM", "KNN"]

    all_model_data = [
    compute_learning_curve_data(rf_model, X_train_main, y_train_main),
    compute_learning_curve_data(lr_model, X_train_main, y_train_main),
    compute_learning_curve_data(svm_model, X_train_main, y_train_main),
    compute_learning_curve_data(knn_model, X_train_main, y_train_main)
]
    # Compute confusion matrices for the first four models (as an example)
    cm_rf = compute_confusion_matrix(y_test_main, y_pred_rf, all_labels)
    cm_lr = compute_confusion_matrix(y_test_main, y_pred_lr, all_labels)
    cm_svm = compute_confusion_matrix(y_test_main, y_pred_svm, all_labels)
    cm_knn = compute_confusion_matrix(y_test_main, y_pred_knn, all_labels)

    confusion_matrices = [cm_rf, cm_lr, cm_svm, cm_knn]
    fig_lc, axes_lc = plot_2x2_learning_curves(
            all_model_data,
            model_names_for_curves,
            save_path=os.path.join(PLOT_SAVE_DIR, "2x2_learning_curves.png")
        )

        # 3) 2x2 Precision-Recall curves
    fig_pr, axes_pr = plot_2x2_precision_recall_curves(
        [rf_model, lr_model, svm_model, knn_model],
        model_names_for_curves,
        X_test_main,
        y_test_main,
        label_encoder=label_encoder_main, 
        save_path=os.path.join(PLOT_SAVE_DIR, "2x2_precision_recall_curves.png")
    )

    # 2) 2x2 ROC curves
    fig_roc, axes_roc = plot_2x2_roc_curves(
        [rf_model, lr_model, svm_model, knn_model],
        model_names_for_curves,
        X_test_main,
        y_test_main,
        label_encoder=label_encoder_main,  
        save_path=os.path.join(PLOT_SAVE_DIR, "2x2_roc_curves.png")
    )

    model_names = ["RandomForest", "LogisticRegression", "SVM", "KNN"]
    # Plotting Confusion Matrices
    fig_cm, axes_cm = plot_2x2_confusion_matrices(
        confusion_matrices=confusion_matrices,
        model_names=model_names,
        all_labels=all_labels,
        normalize=False,  # Set to True if you want normalized confusion matrices
        save_path=os.path.join(PLOT_SAVE_DIR, "2x2_confusion_matrices.png")
    )

    



    
logging.info("Done with main_type pipeline!")


if __name__ == "__main__":
    main_type()
    """
    compute_multiclass_roc_curves(
        models=models_for_curves,
        model_names=model_names_for_curves,
        X_test=X_test_main,
        y_test=y_test_main,
        labels=label_encoder_main.classes_,
        dataset_name="test_main",
        predictor_type="flat",
    )

    compute_multiclass_precision_recall_curves(
        models=models_for_curves,
        model_names=model_names_for_curves,
        X_test=X_test_main,
        y_test=y_test_main,
        labels=label_encoder_main.classes_,
        dataset_name="test_main",
        predictor_type="flat",
    )


    compute_confusion_matrix(
        lr_model, X_test, y_test_encoded, all_labels,
        dataset_name="test", predictor_type="flat", model_name="LogisticRegression",
        label_encoder=label_encoder
    )

    compute_confusion_matrix(
        svm_model, X_test, y_test_encoded, all_labels,
        dataset_name="test", predictor_type="flat", model_name="SVM",
        label_encoder=label_encoder
    )

    compute_confusion_matrix(
        knn_model, X_test, y_test_encoded, all_labels,
        dataset_name="test", predictor_type="flat", model_name="KNN",
        label_encoder=label_encoder
    )
    logging.info("Computing learning curves...")
    compute_learning_curve(rf_model, X_train_main, y_train_main, "train_main", "flat", "RandomForest")
    compute_learning_curve(lr_model, X_train_main, y_train_main, "train_main", "flat", "LogisticRegression")
    compute_learning_curve(svm_model, X_train_main, y_train_main, "train_main", "flat", "SVM")
    compute_learning_curve(knn_model, X_train_main, y_train_main, "train_main", "flat", "KNN")

"""