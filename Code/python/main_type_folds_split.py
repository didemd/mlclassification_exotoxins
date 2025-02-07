import logging
import os
import numpy as np
import pandas as pd

from config.config import PLOT_SAVE_DIR, TRAINING_EMBEDDINGS_PATH_FOLDS 
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
    generate_class_metrics_table,
    
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

from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main_type():
    """
    Main pipeline for 'Type' prediction, including hierarchical classifiers, 
    BLAST integration, confusion matrices, MCC comparisons, etc.
    """
    # ----------------------------------------------------------------------
    training_labels_path = './data/ToxinTypes_labelTarget_3.csv'
    blast_results_path = './data/blast_results.tsv'
    model_save_dir = './plots'

    # ----------------------------------------------------------------------
    # 3. Load and merge training data
    # ----------------------------------------------------------------------
    logging.info("Loading Training Embeddings...")
    training_embeddings_df = load_embeddings(TRAINING_EMBEDDINGS_PATH_FOLDS)

    if training_embeddings_df is None:
        logging.error("Failed to load training embeddings.")
        return

    logging.info("Loading Training Labels...")
    training_labels_df = load_labels(training_labels_path)
    print("Columns in training_labels_df after loading:", training_labels_df.columns.tolist())
    if training_labels_df is None:
        logging.error("Failed to load training labels.")
        return

    logging.info("Merging Training Embeddings and Labels...")
    merged_train_df = merge_embeddings_labels(training_embeddings_df, training_labels_df)
    print("Columns in merged_train_df:", merged_train_df.columns.tolist())

    if merged_train_df is None or merged_train_df.empty:
        logging.error("Failed to merge or no data after merging.")
        return

    # ----------------------------------------------------------------------
    # 4. (Optional) Filter out "Unknown" before splitting
    # ----------------------------------------------------------------------
    logging.info("Filtering out 'Unknown' labels if present...")
    all_labels = ['Type_I', 'Type_II', 'Type_III', 'Type_IV','Unknown'] 
    merged_train_df = merged_train_df[merged_train_df['type'].isin(all_labels)].copy()
    print(merged_train_df)
    if merged_train_df.empty:
        logging.error("No data left after filtering out 'Unknown' labels.")
        return

    # ----------------------------------------------------------------------
    # 5. Split Data into Training and Test Sets
    # ----------------------------------------------------------------------
    logging.info("Splitting data into training and test sets with an 80-20 ratio...")
    train_df_main, test_df_main = train_test_split(
        merged_train_df,
        test_size=0.2,
        random_state=42,
        stratify=merged_train_df['type']
    )

    logging.info(f"Training set size: {train_df_main.shape[0]} samples")
    logging.info(f"Test set size: {test_df_main.shape[0]} samples")

    # ----------------------------------------------------------------------
    # 6. Preprocess Main Training and Test Data
    # ----------------------------------------------------------------------
    logging.info("Preprocessing Training Data...")
    try:
        X_train_main, y_train_main, label_encoder_main = preprocess_data(train_df_main)
    except ValueError as ve:
        logging.error(f"Preprocessing training data failed: {ve}")
        return

    logging.info("Preprocessing Test Data...")
    try:
        X_test_main, y_test_main, _ = preprocess_data(test_df_main)
    except ValueError as ve:
        logging.error(f"Preprocessing test data failed: {ve}")
        return

    # Define feature columns if necessary
    # Assuming preprocess_data returns numpy arrays, feature_columns_main may not be needed
    # If feature columns are required for other operations, define them here
    # For example:
    # feature_columns_main = [col for col in merged_train_df.columns if col not in ['type', 'other_non_feature_columns']]

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    logging.info("Applying label masks to training and test data...")
    train_mask = train_df_main['type'].isin(all_labels)
    train_df_main = train_df_main[train_mask].copy()

    test_mask = test_df_main['type'].isin(all_labels)
    test_df_main = test_df_main[test_mask].copy()

    # Re-encode after filtering
    label_encoder_main = LabelEncoder()
    label_encoder_main.fit(all_labels)
    y_train_main = label_encoder_main.transform(train_df_main['type'])
    y_test_main = label_encoder_main.transform(test_df_main['type'])

    # ----------------------------------------------------------------------
    # 7. Train hierarchical classifiers
    # ----------------------------------------------------------------------
    logging.info("Training Hierarchical RF Classifier...")
    model_rf_hier = train_hierarchical_classifier_rf(train_df_main)

    logging.info("Training Hierarchical SVM Classifier...")
    model_svm_hier = train_hierarchical_classifier_svm(train_df_main)

    # ----------------------------------------------------------------------
    # 8. Train "flat" classifiers
    # ----------------------------------------------------------------------
    n_components = 50
    logging.info("Training Random Forest...")
    rf_model, rf_grid_search, _ = train_random_forest(X_train_main, y_train_main, n_components)

    logging.info("Training Logistic Regression...")
    lr_model, lr_grid_search, _ = train_logistic_regression(X_train_main, y_train_main, n_components)

    logging.info("Training SVM...")
    svm_model, svm_grid_search, _ = train_svm(X_train_main, y_train_main, n_components)

    logging.info("Training KNN...")
    knn_model, knn_grid_search, _ = train_k_neighbors(X_train_main, y_train_main, n_components)

    # ----------------------------------------------------------------------
    # 9. Predictions
    # ----------------------------------------------------------------------
    logging.info("Evaluating Hierarchical RF on Test Set...")
    y_pred_rf_hier_str = model_rf_hier.predict(X_test_main)
    y_pred_rf_hier_encoded = label_encoder_main.transform(y_pred_rf_hier_str)

    logging.info("Evaluating Hierarchical SVM on Test Set...")
    y_pred_svm_hier_str = model_svm_hier.predict(X_test_main)
    y_pred_svm_hier_encoded = label_encoder_main.transform(y_pred_svm_hier_str)

    logging.info("Evaluating Flat Random Forest...")
    y_pred_rf = rf_model.predict(X_test_main)

    logging.info("Evaluating Flat Logistic Regression...")
    y_pred_lr = lr_model.predict(X_test_main)

    logging.info("Evaluating Flat SVM...")
    y_pred_svm = svm_model.predict(X_test_main)

    logging.info("Evaluating Flat KNN...")
    y_pred_knn = knn_model.predict(X_test_main)

    # ----------------------------------------------------------------------
    # 10. Evaluate all classifiers
    # ----------------------------------------------------------------------
    hier_rf_metrics = calculate_evaluation_metrics(
        y_test_main, y_pred_rf_hier_encoded, label_encoder=label_encoder_main
    )
    table_hier_rf = generate_table(hier_rf_metrics)
    print("\nHierarchical RF Metrics:\n", table_hier_rf)
    save_metrics_to_csv(hier_rf_metrics, "Hierarchical_RF_metrics.csv")
    save_table_to_file(table_hier_rf, "Hierarchical_RF_metrics_table.txt")

    hier_rf_class_metrics = generate_class_metrics_table(y_test_main, y_pred_rf_hier_encoded, all_labels)
    print("\n Hier RF (Flat) Per-Class Metrics Table:\n", hier_rf_class_metrics)
    hier_rf_class_metrics.to_csv(os.path.join(PLOT_SAVE_DIR, "Hier_RF_flat_per_class_metrics.csv"), index=False)

    hier_svm_metrics = calculate_evaluation_metrics(
        y_test_main, y_pred_svm_hier_encoded, label_encoder=label_encoder_main
    )
    table_hier_svm = generate_table(hier_svm_metrics)
    print("\nHierarchical SVM Metrics:\n", table_hier_svm)
    save_metrics_to_csv(hier_svm_metrics, "Hierarchical_SVM_metrics.csv")
    save_table_to_file(table_hier_svm, "Hierarchical_SVM_metrics_table.txt")

    hier_svm_class_metrics = generate_class_metrics_table(y_test_main, y_pred_svm_hier_encoded, all_labels)
    print("\n Hier SVM (Flat) Per-Class Metrics Table:\n", hier_svm_class_metrics)
    hier_svm_class_metrics.to_csv(os.path.join(PLOT_SAVE_DIR, "Hier_SVM_flat_per_class_metrics.csv"), index=False)

    rf_metrics = calculate_evaluation_metrics(y_test_main, y_pred_rf, label_encoder=label_encoder_main)
    table_rf = generate_table(rf_metrics)
    print("\nRandom Forest (flat) Metrics:\n", table_rf)
    save_metrics_to_csv(rf_metrics, "RandomForest_flat_metrics.csv")
    save_table_to_file(table_rf, "RandomForest_flat_metrics_table.txt")

    rf_class_metrics = generate_class_metrics_table(y_test_main, y_pred_rf, all_labels)
    print("\nRandom Forest (Flat) Per-Class Metrics Table:\n", rf_class_metrics)
    rf_class_metrics.to_csv(os.path.join(PLOT_SAVE_DIR, "RandomForest_flat_per_class_metrics.csv"), index=False)


    lr_metrics = calculate_evaluation_metrics(y_test_main, y_pred_lr, label_encoder=label_encoder_main)
    table_lr = generate_table(lr_metrics)
    print("\nLogistic Regression (flat) Metrics:\n", table_lr)
    save_metrics_to_csv(lr_metrics, "LogisticRegression_flat_metrics.csv")
    save_table_to_file(table_lr, "LogisticRegression_flat_metrics_table.txt")

    lr_class_metrics = generate_class_metrics_table(y_test_main, y_pred_lr, all_labels)
    print("\nLogistic Regression (Flat) Per-Class Metrics Table:\n", lr_class_metrics)
    lr_class_metrics.to_csv(os.path.join(PLOT_SAVE_DIR, "LogisticRegression_flat_per_class_metrics.csv"), index=False)


    svm_metrics = calculate_evaluation_metrics(y_test_main, y_pred_svm, label_encoder=label_encoder_main)
    table_svm = generate_table(svm_metrics)
    print("\nSVM (flat) Metrics:\n", table_svm)
    save_metrics_to_csv(svm_metrics, "SVM_flat_metrics.csv")
    save_table_to_file(table_svm, "SVM_flat_metrics_table.txt")

    svm_class_metrics = generate_class_metrics_table(y_test_main, y_pred_svm, all_labels)
    print("\n SVM (Flat) Per-Class Metrics Table:\n", svm_class_metrics)
    svm_class_metrics.to_csv(os.path.join(PLOT_SAVE_DIR, "SVM_flat_per_class_metrics.csv"), index=False)

    knn_metrics = calculate_evaluation_metrics(y_test_main, y_pred_knn, label_encoder=label_encoder_main)
    table_knn = generate_table(knn_metrics)
    print("\nKNN (flat) Metrics:\n", table_knn)
    save_metrics_to_csv(knn_metrics, "KNN_flat_metrics.csv")
    save_table_to_file(table_knn, "KNN_flat_metrics_table.txt")

    knn_class_metrics = generate_class_metrics_table(y_test_main, y_pred_knn, all_labels)
    print("\n SVM (Flat) Per-Class Metrics Table:\n", knn_class_metrics)
    knn_class_metrics.to_csv(os.path.join(PLOT_SAVE_DIR, "KNN_flat_per_class_metrics.csv"), index=False)


    # ----------------------------------------------------------------------
    # 12. Combine MCC & SE in a DataFrame, then plot
    # ----------------------------------------------------------------------
    logging.info("Aggregating MCC and SE for all predictors...")

    all_predictors = [
        "RF",
        "LR",
        "SVM",
        "KNN",
        "Hier_RF",
        "Hier_SVM"
    ]

    metrics_dict = {
        "RF": rf_metrics,
        "LR": lr_metrics,
        "SVM": svm_metrics,
        "KNN": knn_metrics,
        "Hier_RF": hier_rf_metrics,
        "Hier_SVM": hier_svm_metrics,
    }

    data = {"Predictor": [], "MCC": [], "MCC_SE": []}
    for name in all_predictors:
        mcc_val, mcc_se = metrics_dict.get(name, {}).get('MCC', (0.0, 0.0))
        data["Predictor"].append(name)
        data["MCC"].append(mcc_val)
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

    plot_comparative_bar_graph(
        combined_mcc_df,
        bar_colors=bar_colors,
        save_path=os.path.join(PLOT_SAVE_DIR, "comparison_graph_with_error_bars.png")
    )

    # ----------------------------------------------------------------------
    # 13. Learning Curves
    # ----------------------------------------------------------------------
    logging.info("Computing learning curves...")
    # 14. ROC and Precision-Recall curves (Multiclass)
    # ----------------------------------------------------------------------
    logging.info("Computing ROC and Precision-Recall curves...")

    models_for_curves = [rf_model, lr_model, svm_model, knn_model]
    model_names_for_curves = ["RandomForest", "LogisticRegression", "SVM", "KNN"]

    all_model_data = [
    compute_learning_curve_data(rf_model, X_train_main, y_train_main),
    compute_learning_curve_data(lr_model, X_train_main, y_train_main),
    compute_learning_curve_data(svm_model, X_train_main, y_train_main),
    compute_learning_curve_data(knn_model, X_train_main, y_train_main)
]
    # Compute confusion matrices for the first four models (as an example)
    cm_rf= compute_confusion_matrix(y_test_main, y_pred_rf, all_labels)
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