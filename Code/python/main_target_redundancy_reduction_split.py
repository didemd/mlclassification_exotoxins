import logging
import os
import numpy as np
import pandas as pd
import pickle


from config.config import (
    TRAINING_EMBEDDINGS_PATH,
    TEST_EMBEDDINGS_PATH,
    TRAINING_LABELS_PATH,
    TEST_LABELS_PATH,
    ALL_LABELS,
    PLOT_SAVE_DIR,
    STATS_SAVE_DIR,
    N_COMPONENTS,
    MODEL_SAVE_DIR
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
#from evaluation.confusion_matrix_plot import compute_confusion_matrix

from visualization.plots import(
    plot_2x2_learning_curves,
    plot_2x2_roc_curves,
    plot_2x2_precision_recall_curves,
    compute_learning_curve_data,
    compute_confusion_matrix,
    plot_2x2_confusion_matrices,
    generate_class_metrics_table,
    plot_comparative_bar_graph_target
)

from blast.blast_predictor import (
    load_blast_hits,
    load_blast_labels,
    run_blast_predictor,
    compute_confusion_matrix_blast
)

def main_target_redundancy_reduction_split():
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

    # ----------------------------------------------------------------------
    logging.info("Loading Training Embeddings...")
    training_embeddings_df = load_embeddings(TRAINING_EMBEDDINGS_PATH)
    if training_embeddings_df is None:
        logging.error("Failed to load training embeddings.")
        return

    logging.info("Loading Training Labels...")
    training_labels_df = load_labels(TRAINING_LABELS_PATH, selected_column='target')
    if training_labels_df is None:
        logging.error("Failed to load training labels.")
        return

    logging.info("Merging Training Embeddings and Labels...")
    merged_train_df = merge_embeddings_labels(training_embeddings_df, training_labels_df, selected_column='target')
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
    training_df_main['target'] = label_encoder_main.inverse_transform(y_train_main)

    # ----------------------------------------------------------------------
    # 5. Load and merge test data
    # ----------------------------------------------------------------------
    logging.info("Loading Test Embeddings...")
    test_embeddings_df = load_embeddings(TEST_EMBEDDINGS_PATH)
    if test_embeddings_df is None:
        logging.error("Failed to load test embeddings.")
        return

    logging.info("Loading Test Labels...")
    test_labels_df = load_labels(TEST_LABELS_PATH, selected_column='target')
    if test_labels_df is None:
        logging.error("Failed to load test labels.")
        return

    logging.info("Merging Test Embeddings and Labels...")
    merged_test_df = merge_embeddings_labels(test_embeddings_df, test_labels_df, selected_column='target')
    if merged_test_df is None or merged_test_df.empty:
        logging.error("Failed to merge test embeddings and labels.")
        return

    logging.info("Preprocessing Test Data...")
    X_test_main, y_test_main, _ = preprocess_data(merged_test_df)

    test_df_main = pd.DataFrame(X_test_main, columns=feature_columns_main)
    test_df_main['target'] = label_encoder_main.inverse_transform(y_test_main)

    print("LabelEncoder classes:", label_encoder_main.classes_)

    # ----------------------------------------------------------------------
    # 6. Train "Flat" Classifiers
    # ----------------------------------------------------------------------
    n_components = 50  # Adjust based on your data
    logging.info("Training Random Forest...")
    try:
        rf_model, rf_grid_search, _ = train_random_forest(X_train_main, y_train_main, n_components)
    except Exception as e:
        logging.error(f"Training Random Forest failed: {e}")
        return

    logging.info("Training Logistic Regression...")
    try:
        lr_model, lr_grid_search, _ = train_logistic_regression(X_train_main, y_train_main, n_components)
    except Exception as e:
        logging.error(f"Training Logistic Regression failed: {e}")
        return

    logging.info("Training SVM...")
    try:
        svm_model, svm_grid_search, _ = train_svm(X_train_main, y_train_main, n_components)
    except Exception as e:
        logging.error(f"Training SVM failed: {e}")
        return

    logging.info("Training KNN...")
    try:
        knn_model, knn_grid_search, _ = train_k_neighbors(X_train_main, y_train_main, n_components)
    except Exception as e:
        logging.error(f"Training KNN failed: {e}")
        return

    # ----------------------------------------------------------------------
    # 7. Predictions
    # ----------------------------------------------------------------------
    logging.info("Evaluating Flat Random Forest...")
    try:
        y_pred_rf = rf_model.predict(X_test_main)
    except Exception as e:
        logging.error(f"Evaluating Flat Random Forest failed: {e}")
        return

    logging.info("Evaluating Flat Logistic Regression...")
    try:
        y_pred_lr = lr_model.predict(X_test_main)
    except Exception as e:
        logging.error(f"Evaluating Flat Logistic Regression failed: {e}")
        return

    logging.info("Evaluating Flat SVM...")
    try:
        y_pred_svm = svm_model.predict(X_test_main)
    except Exception as e:
        logging.error(f"Evaluating Flat SVM failed: {e}")
        return

    logging.info("Evaluating Flat KNN...")
    try:
        y_pred_knn = knn_model.predict(X_test_main)
    except Exception as e:
        logging.error(f"Evaluating Flat KNN failed: {e}")
        return

    # ----------------------------------------------------------------------
    # 8. Evaluate All Classifiers
    # ----------------------------------------------------------------------
    logging.info("Calculating evaluation metrics for all classifiers...")
    try:
        rf_metrics = calculate_evaluation_metrics(y_test_main, y_pred_rf, label_encoder=label_encoder_main)
        table_rf = generate_table(rf_metrics)
        print("\nRandom Forest (flat) Metrics:\n", table_rf)
        save_metrics_to_csv(rf_metrics, os.path.join(STATS_SAVE_DIR, "RandomForest_flat_metrics.csv"))
        save_table_to_file(table_rf, os.path.join(STATS_SAVE_DIR, "RandomForest_flat_metrics_table.txt"))

        rf_class_metrics = generate_class_metrics_table(y_test_main, y_pred_rf, ALL_LABELS)
        print("\nRandom Forest (Flat) Per-Class Metrics Table:\n", rf_class_metrics)
        rf_class_metrics.to_csv(os.path.join(STATS_SAVE_DIR, "RandomForest_flat_per_class_metrics.csv"), index=False)

        # Logistic Regression Metrics
        lr_metrics = calculate_evaluation_metrics(y_test_main, y_pred_lr, label_encoder=label_encoder_main)
        table_lr = generate_table(lr_metrics)
        print("\nLogistic Regression (flat) Metrics:\n", table_lr)
        save_metrics_to_csv(lr_metrics, os.path.join(STATS_SAVE_DIR, "LogisticRegression_flat_metrics.csv"))
        save_table_to_file(table_lr, os.path.join(STATS_SAVE_DIR, "LogisticRegression_flat_metrics_table.txt"))

        lr_class_metrics = generate_class_metrics_table(y_test_main, y_pred_lr, ALL_LABELS)
        print("\nLogistic Regression (Flat) Per-Class Metrics Table:\n", lr_class_metrics)
        lr_class_metrics.to_csv(os.path.join(STATS_SAVE_DIR, "LogisticRegression_flat_per_class_metrics.csv"), index=False)

        # SVM Metrics
        svm_metrics = calculate_evaluation_metrics(y_test_main, y_pred_svm, label_encoder=label_encoder_main)
        table_svm = generate_table(svm_metrics)
        print("\nSVM (flat) Metrics:\n", table_svm)
        save_metrics_to_csv(svm_metrics, os.path.join(STATS_SAVE_DIR, "SVM_flat_metrics.csv"))
        save_table_to_file(table_svm, os.path.join(STATS_SAVE_DIR, "SVM_flat_metrics_table.txt"))

        svm_class_metrics = generate_class_metrics_table(y_test_main, y_pred_svm, ALL_LABELS)
        print("\nSVM (Flat) Per-Class Metrics Table:\n", svm_class_metrics)
        svm_class_metrics.to_csv(os.path.join(STATS_SAVE_DIR, "SVM_flat_per_class_metrics.csv"), index=False)

        # KNN Metrics
        knn_metrics = calculate_evaluation_metrics(y_test_main, y_pred_knn, label_encoder=label_encoder_main)
        table_knn = generate_table(knn_metrics)
        print("\nKNN (flat) Metrics:\n", table_knn)
        save_metrics_to_csv(knn_metrics, os.path.join(STATS_SAVE_DIR, "KNN_flat_metrics.csv"))
        save_table_to_file(table_knn, os.path.join(STATS_SAVE_DIR, "KNN_flat_metrics_table.txt"))

        knn_class_metrics = generate_class_metrics_table(y_test_main, y_pred_knn, ALL_LABELS)
        print("\nKNN (Flat) Per-Class Metrics Table:\n", knn_class_metrics)
        knn_class_metrics.to_csv(os.path.join(STATS_SAVE_DIR, "KNN_flat_per_class_metrics.csv"), index=False)

    except Exception as e:
        logging.error(f"An error occurred during evaluation: {e}")
        return

    # ----------------------------------------------------------------------
    # 10. Combine MCC & SE in a DataFrame, Then Plot
    # ----------------------------------------------------------------------
    logging.info("Aggregating MCC and SE for all predictors...")

    all_predictors = [
        "RF",
        "LR",
        "SVM",
        "KNN",
    ]

    metrics_dict = {
        "RF": rf_metrics,
        "LR": lr_metrics,
        "SVM": svm_metrics,
        "KNN": knn_metrics,
    }

    data = {"Predictor": [], "MCC": [], "MCC_SE": []}
    for name in all_predictors:
        mcc_val, mcc_se = metrics_dict.get(name, {}).get('MCC', (0.0, 0.0))
        data["Predictor"].append(name)
        data["MCC"].append(mcc_val)
        data["MCC_SE"].append(mcc_se)

    combined_mcc_df = pd.DataFrame(data)
    logging.info(f"Combined MCC DataFrame:\n{combined_mcc_df}")

    bar_colors =  [
    "#882255",  # Dark Red-Purple
    "#44AA99",  # Teal
    "#117733",  # Dark Green
    "#332288",  # Dark Blue
    "#DDCC77",  # Tan/Gold
    "#88CCEE"   # Cyan
]

    try:
        plot_comparative_bar_graph_target(
            combined_mcc_df,
            bar_colors=bar_colors,
            save_path=os.path.join(PLOT_SAVE_DIR, "comparison_graph_with_error_bars.png")
        )
    except Exception as e:
        logging.error(f"Plotting comparative bar graph failed: {e}")
        return

    # ----------------------------------------------------------------------
    # 11. Learning Curves
    # ----------------------------------------------------------------------
    logging.info("Computing learning curves...")
    try:
        all_model_data = [
            compute_learning_curve_data(rf_model, X_train_main, y_train_main),
            compute_learning_curve_data(lr_model, X_train_main, y_train_main),
            compute_learning_curve_data(svm_model, X_train_main, y_train_main),
            compute_learning_curve_data(knn_model, X_train_main, y_train_main)
        ]

        model_names_for_curves = ["RandomForest", "LogisticRegression", "SVM", "KNN"]

        fig_lc, axes_lc = plot_2x2_learning_curves(
            all_model_data,
            model_names_for_curves,
            save_path=os.path.join(PLOT_SAVE_DIR, "2x2_learning_curves.png")
        )
    except Exception as e:
        logging.error(f"Computing learning curves failed: {e}")
        return

    # ----------------------------------------------------------------------
    # 12. ROC and Precision-Recall Curves (Multiclass)
    # ----------------------------------------------------------------------
    logging.info("Computing ROC and Precision-Recall curves...")
    try:
        fig_roc, axes_roc = plot_2x2_roc_curves(
            [rf_model, lr_model, svm_model, knn_model],
            model_names_for_curves,
            X_test_main,
            y_test_main,
            label_encoder=label_encoder_main,  
            save_path=os.path.join(PLOT_SAVE_DIR, "2x2_roc_curves.png")
        )

        fig_pr, axes_pr = plot_2x2_precision_recall_curves(
            [rf_model, lr_model, svm_model, knn_model],
            model_names_for_curves,
            X_test_main,
            y_test_main,
            label_encoder=label_encoder_main, 
            save_path=os.path.join(PLOT_SAVE_DIR, "2x2_precision_recall_curves.png")
        )
    except Exception as e:
        logging.error(f"Computing ROC and Precision-Recall curves failed: {e}")
        return

    # ----------------------------------------------------------------------
    # 13. Confusion Matrices
    # ----------------------------------------------------------------------
    logging.info("Computing confusion matrices for flat classifiers...")
    try:
        cm_rf = compute_confusion_matrix(y_test_main, y_pred_rf, ALL_LABELS)
        cm_lr = compute_confusion_matrix(y_test_main, y_pred_lr, ALL_LABELS)
        cm_svm = compute_confusion_matrix(y_test_main, y_pred_svm, ALL_LABELS)
        cm_knn = compute_confusion_matrix(y_test_main, y_pred_knn, ALL_LABELS)

        confusion_matrices = [cm_rf, cm_lr, cm_svm, cm_knn]
        model_names = ["RandomForest", "LogisticRegression", "SVM", "KNN"]

        fig_cm, axes_cm = plot_2x2_confusion_matrices(
            confusion_matrices=confusion_matrices,
            model_names=model_names,
            all_labels=ALL_LABELS,
            normalize=False,  # Set to True if you want normalized confusion matrices
            save_path=os.path.join(PLOT_SAVE_DIR, "2x2_confusion_matrices.png")
        )
    except Exception as e:
        logging.error(f"Computing confusion matrices failed: {e}")
        return

    # Save trained models to MODEL_SAVE_DIR
    logging.info("Saving trained models...")
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        logging.info(f"Created predictor save directory at {MODEL_SAVE_DIR}")
    
    try:        
        # Save models with descriptive names
        with open(os.path.join(MODEL_SAVE_DIR, "random_forest_model.pkl"), 'wb') as f:
            pickle.dump(rf_model, f)
        
        with open(os.path.join(MODEL_SAVE_DIR, "logistic_regression_model.pkl"), 'wb') as f:
            pickle.dump(lr_model, f)
        
        with open(os.path.join(MODEL_SAVE_DIR, "svm_model.pkl"), 'wb') as f:
            pickle.dump(svm_model, f)
        
        with open(os.path.join(MODEL_SAVE_DIR, "knn_model.pkl"), 'wb') as f:
            pickle.dump(knn_model, f)
        
        logging.info(f"All models successfully saved to {MODEL_SAVE_DIR}")
    except Exception as e:
        logging.error(f"Failed to save models: {e}")

    logging.info("Done with main_target_redundancy_reduction_split pipeline!")

if __name__ == "__main__":
    main_target_redundancy_reduction_split()