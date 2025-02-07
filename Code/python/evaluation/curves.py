import logging
import os
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    average_precision_score,
    make_scorer,
    matthews_corrcoef,
)
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.model_selection import learning_curve
from config.config import PLOT_SAVE_DIR
from visualization.plots import save_plot

def compute_learning_curve(model, X_train, y_train, dataset_name, predictor_type, model_name):
    """
    Compute and plot the learning curve for a given model.
    """
    if not hasattr(model, 'fit'):
        logging.warning(f"Model {model_name} does not implement 'fit'. Skipping learning curve computation.")
        return

    try:
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=model,
            X=X_train,
            y=y_train,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5,
            n_jobs=-1,
            scoring=make_scorer(matthews_corrcoef)
        )
        logging.info("Learning curve computed")

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        fig = plot_learning_curve(train_sizes, train_mean, train_std, test_mean, test_std, model_name)
        plot_filename = f'learning_curve_{dataset_name}_{predictor_type}_{model_name}.png'
        save_plot(fig, plot_filename)

    except Exception as e:
        logging.error(f"An error occurred: {e}")

def compute_roc_curves(models, model_names, X_test, y_test, labels, dataset_name, predictor_type):
    """
    Compute and plot the ROC curves for multiple models.
    """
    try:
        fig = plot_roc_curve(models, model_names, X_test, y_test)
        plot_filename = f'roc_curves_comparison_{dataset_name}_{predictor_type}.png'
        save_plot(fig, plot_filename)
    except Exception as e:
        logging.error(f"An error occurred while computing ROC curves: {e}")

def compute_precision_recall_curve(models, model_names, X_test, y_test, labels, dataset_name, predictor_type):
    """
    Compute and plot the Precision-Recall curves for multiple models.
    """
    try:
        fig = plot_precision_recall_curve(models, model_names, X_test, y_test)
        plot_filename = f'precision_recall_curves_comparison_{dataset_name}_{predictor_type}.png'
        save_plot(fig, plot_filename)
    except Exception as e:
        logging.error(f"An error occurred while computing Precision-Recall curves: {e}")

def compute_multiclass_roc_curves(models, model_names, X_test, y_test, labels, dataset_name, predictor_type):
    """
    Compute and plot multi-class ROC curves for multiple models using One-vs-Rest strategy.

    Args:
        models (list): List of trained classifier models.
        model_names (list): List of model names corresponding to the models.
        X_test (array-like): Test feature set.
        y_test (array-like): True labels for the test set.
        labels (list): List of unique class labels.
        dataset_name (str): Name of the dataset (e.g., 'test').
        predictor_type (str): Type of predictor (e.g., 'flat', 'hierarchical').
        save_plot (function): Function to save the plot. Should accept (fig, filename).

    Returns:
        None
    """
    try:
        # Binarize the output labels for multi-class ROC
        y_test_binarized = label_binarize(y_test, classes=labels)
        n_classes = y_test_binarized.shape[1]

        # Define a color cycle
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink'])


        fig = plt.figure(figsize=(12, 8))  # create a new figure
        ax = fig.add_subplot(111)         # or you can use plt.subplots()

        for model, model_name in zip(models, model_names):
            if not hasattr(model, "predict_proba"):
                logging.warning(f"Model '{model_name}' does not support probability predictions. Skipping ROC curve.")
                continue

            # Get prediction probabilities
            y_score = model.predict_proba(X_test)

            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i, color in zip(range(n_classes), colors):
                fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                         label=f'ROC curve of class {labels[i]} for {model_name} (AUC = {roc_auc[i]:0.2f})')

            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            plt.plot(fpr["micro"], tpr["micro"],
                     label=f'micro-average ROC for {model_name} (AUC = {roc_auc["micro"]:0.2f})',
                     color='deeppink', linestyle=':', linewidth=4)

        # Plot the Chance Line
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title(f'Multi-class ROC Curves Comparison ({predictor_type.capitalize()} Predictors)', fontsize=16)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)

        # Save the plot using the provided save_plot function
        plot_filename = f'roc_curves_comparison_{dataset_name}_{predictor_type}.png'
        save_plot(fig, plot_filename) # Save the plot

        plt.close(fig)  # Or fig.close() in newer Matplotlib

    except Exception as e:
        logging.error(f"An error occurred while computing multi-class ROC curves: {e}")

def compute_multiclass_precision_recall_curves(models, model_names, X_test, y_test, labels, dataset_name, predictor_type):
    """
    Compute and plot multi-class Precision-Recall curves for multiple models using One-vs-Rest strategy.

    Args:
        models (list): List of trained classifier models.
        model_names (list): List of model names corresponding to the models.
        X_test (array-like): Test feature set.
        y_test (array-like): True labels for the test set.
        labels (list): List of unique class labels.
        dataset_name (str): Name of the dataset (e.g., 'test').
        predictor_type (str): Type of predictor (e.g., 'flat', 'hierarchical').
        save_plot (function): Function to save the plot. Should accept (fig, filename).

    Returns:
        None
    """
    try:
        # Binarize the output labels for multi-class PR
        y_test_binarized = label_binarize(y_test, classes=labels)
        n_classes = y_test_binarized.shape[1]

        # Define a color cycle
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink'])

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)

        for model, model_name in zip(models, model_names):
            if not hasattr(model, "predict_proba"):
                logging.warning(f"Model '{model_name}' does not support probability predictions. Skipping Precision-Recall curve.")
                continue

            # Get prediction probabilities
            y_score = model.predict_proba(X_test)

            # Compute Precision-Recall curve and average precision for each class
            precision = dict()
            recall = dict()
            average_precision = dict()
            for i, color in zip(range(n_classes), colors):
                precision[i], recall[i], _ = precision_recall_curve(y_test_binarized[:, i], y_score[:, i])
                average_precision[i] = average_precision_score(y_test_binarized[:, i], y_score[:, i])
                plt.plot(recall[i], precision[i], color=color, lw=2,
                         label=f'PR curve of class {labels[i]} for {model_name} (AP = {average_precision[i]:0.2f})')

            # Compute micro-average Precision-Recall curve and average precision
            precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_binarized.ravel(), y_score.ravel())
            average_precision["micro"] = average_precision_score(y_test_binarized, y_score, average="micro")
            plt.plot(recall["micro"], precision["micro"],
                     label=f'micro-average PR for {model_name} (AP = {average_precision["micro"]:0.2f})',
                     color='gold', linestyle=':', linewidth=4)

        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title(f'Multi-class Precision-Recall Curves Comparison ({predictor_type.capitalize()} Predictors)', fontsize=16)
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(alpha=0.3)

        # Save the plot using the provided save_plot function
        plot_filename = f'precision_recall_curves_comparison_{dataset_name}_{predictor_type}.png'
        save_plot(fig, plot_filename)

        plt.close()

    except Exception as e:
        logging.error(f"An error occurred while computing multi-class Precision-Recall curves: {e}")