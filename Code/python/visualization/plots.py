import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from config.config import PLOT_SAVE_DIR

import numpy as np
import pandas as pd

plt.rcParams.update({
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

def apply_common_grid(ax, which='both', linestyle='--', linewidth=0.7, color='gray'):
    """
    Apply a common grid style to the given Axes object.
    
    Parameters:
        ax (matplotlib.axes.Axes): The axes to apply the grid to.
        which (str): 'major', 'minor', or 'both'. Determines which grid lines to show.
        linestyle (str): The style of the grid lines.
        linewidth (float): The width of the grid lines.
        color (str): The color of the grid lines.
    """
    ax.grid(
        visible=True, 
        which=which, 
        linestyle=linestyle, 
        linewidth=linewidth, 
        color=color,
        axis='both'
    )
    ax.set_axisbelow(True)  # Ensure grid lines are below other plot elements

import matplotlib.pyplot as plt
import numpy as np

def plot_comparative_bar_graph(combined_mcc_df, bar_colors=None, save_path=None):
    """
    Plot a bar chart of MCC ± MCC_SE for each row in combined_mcc_df.
    
    Expects DataFrame with columns:
       Predictor |  MCC  |  MCC_SE
       
    Parameters:
        combined_mcc_df (DataFrame): The input DataFrame.
        bar_colors (list): A list of colors for each bar. If None, default colors are used.
        save_path (str): Path to save the plot. If None, the plot is not saved.
    """
    # Convert DataFrame columns to arrays
    labels = combined_mcc_df["Predictor"].astype(str).values
    mcc_vals = combined_mcc_df["MCC"].values
    mcc_errs = combined_mcc_df["MCC_SE"].values
    x = np.arange(len(labels))  # X positions for bars

    # Define custom names for X-axis labels
    custom_labels = {
        "RandomForest": "RF",
        "LogisticRegression": "LR",
        "SVM": "SVM",
        "KNN": "KNN",
        "Hierarchical_RF": "Hier_RF",
        "Hierarchical_SVM": "Hier_SVM",
        "BLAST": "BLAST"
    }

    # Use specified bar colors or default
    if bar_colors is None:
        bar_colors = ["skyblue"] * len(labels)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Create bar plot with error bars
    bars = ax.bar(
        x,
        mcc_vals,
        yerr=mcc_errs,
        capsize=5,
        color=bar_colors,
        edgecolor="black"
    )

    # Customize x-axis labels (rename them)
    ax.set_xticks(x)
    ax.set_xticklabels([custom_labels.get(label, label) for label in labels], ha="center")

    # Set axis labels and limits
    ax.set_ylabel("MCC Score")
    ax.set_xlabel("ML models")
    ax.set_ylim([0, 1.0])

    # Annotate each bar with numerical values (MCC ± MCC_SE)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        err = mcc_errs[i]
        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            height + err + 0.02, 
            f"{err:.2f}",
            ha="center", 
        )

    # Apply grid lines for readability
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Bar plot saved to {save_path}")

    plt.show()

def save_plot(fig, plot_filename):
    plot_path = os.path.join(PLOT_SAVE_DIR, plot_filename)
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    logging.info(f"Plot saved as {plot_path}")
    plt.close(fig)

def plot_comparative_bar_graph_target(
    combined_mcc_df, bar_colors=None, save_path=None
):
    """
    Plot a bar chart of MCC ± MCC_SE for each row in combined_mcc_df.
    
    Expects DataFrame with columns:
       Predictor |  MCC  |  MCC_SE
       
    Parameters:
        combined_mcc_df (DataFrame): The input DataFrame.
        bar_colors (list): A list of colors for each bar. If None, default colors are used.
        save_path (str): Path to save the plot. If None, the plot is not saved.
    """

    # Convert DataFrame columns to arrays
    labels = combined_mcc_df["Predictor"].astype(str).values
    mcc_vals = combined_mcc_df["MCC"].values
    mcc_errs = combined_mcc_df["MCC_SE"].values
    x = np.arange(len(labels))  # X positions for bars

    # Define custom names for X-axis labels
    custom_labels = {
        "RandomForest": "RF",
        "LogisticRegression": "LR",
        "SVM": "SVM",
        "KNN": "KNN",
    }

    # Use specified bar colors or default
    if bar_colors is None:
        bar_colors = ["skyblue"] * len(labels)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Create bar plot with error bars
    bars = ax.bar(
        x,
        mcc_vals,
        yerr=mcc_errs,
        capsize=5,
        color=bar_colors,
        edgecolor="black"
    )

    # Customize x-axis labels (rename them)
    ax.set_xticks(x)
    ax.set_xticklabels([custom_labels.get(label, label) for label in labels], ha="center")

    # Set Y-axis label and limits
    ax.set_ylabel("MCC Score")
    ax.set_ylim([0, 1.0])

    # Removed legend; now annotate each bar with its error value.
    for i, bar in enumerate(bars):
        height = bar.get_height()
        err = mcc_errs[i]
        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            height + err + 0.02, 
            f"{err:.2f}",
            ha="center", 
            va="bottom"
        )

    # Apply grid lines for readability
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Bar plot saved to {save_path}")

    plt.show()

def apply_common_grid(ax, which='both', linestyle='--', linewidth=0.7, color='gray'):
    """
    Apply a common grid style to the given Axes object.
    """
    ax.grid(
        visible=True, 
        which=which, 
        linestyle=linestyle, 
        linewidth=linewidth, 
        color=color,
        axis='both'
    )
    ax.set_axisbelow(True)


###############################################################################
# 1) MERGED LEARNING CURVE PLOT
###############################################################################
def plot_merged_learning_curves(
    models, 
    model_names,
    X_train,
    y_train,
    cv=5,
    scoring="matthews_corrcoef",
    save_path=None
):
    """
    Computes and plots learning curves for multiple models on a single figure.

    Parameters:
    -----------
    models : list
        List of trained model objects (e.g. [rf_model, lr_model, ...]).
    model_names : list of str
        Names corresponding to each model in 'models'.
    X_train : np.array or DataFrame
        Training features.
    y_train : array-like
        Training labels.
    cv : int
        Number of CV folds to use for computing the learning curve.
    scoring : str
        Scoring metric for learning_curve(). Default is 'matthews_corrcoef'.
    save_path : str or None
        File path to save the figure. If None, does not save.

    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    from sklearn.model_selection import learning_curve  # local import is fine

    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model, name in zip(models, model_names):
        # 1) Compute learning_curve for this model
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=model,
            X=X_train,
            y=y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,         # Use all CPU cores if you want
            train_sizes=np.linspace(0.1, 1.0, 5)
        )

        # 2) Compute means/stdev
        train_mean = np.mean(train_scores, axis=1)
        train_std  = np.std(train_scores, axis=1)
        test_mean  = np.mean(test_scores, axis=1)
        test_std   = np.std(test_scores, axis=1)

        # 3) Plot training curve
        ax.plot(train_sizes, train_mean, marker='o', label=f"{name} - Train")
        ax.fill_between(train_sizes,
                        train_mean - train_std,
                        train_mean + train_std,
                        alpha=0.1)

        # 4) Plot validation curve
        ax.plot(train_sizes, test_mean, marker='s', linestyle='--', label=f"{name} - Val")
        ax.fill_between(train_sizes,
                        test_mean - test_std,
                        test_mean + test_std,
                        alpha=0.1)

    apply_common_grid(ax)
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("MCC")
    ax.set_ylim([-0.05, 1.05])
    ax.set_title("Merged Learning Curves (MCC)")
    ax.legend(loc="best")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Merged Learning Curves saved to {save_path}")

    return fig


###############################################################################
# 2) MERGED ROC CURVE PLOT
###############################################################################
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

###############################################################################
# 3) MERGED PRECISION-RECALL CURVE PLOT
###############################################################################
def plot_merged_precision_recall_curves(
    models,
    model_names,
    X_test,
    y_test,
    label_encoder=None,
    save_path=None
):
    """
    Computes and plots Precision-Recall curves for multiple models 
    on one figure (binary or multiclass one-vs-rest approach).
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Check if multiclass
    if label_encoder is not None and len(label_encoder.classes_) > 2:
        # MULTICLASS
        n_classes = len(label_encoder.classes_)
        y_test_binarized = _binarize_labels(y_test, n_classes)
        
        for model, name in zip(models, model_names):
            if not hasattr(model, "predict_proba"):
                logging.warning(f"Model {name} has no predict_proba; skipping PR curve.")
                continue

            y_score = model.predict_proba(X_test)
            
            # Compute for each class, then average
            precision_dict = {}
            recall_dict    = {}
            average_prec   = []
            
            for i in range(n_classes):
                precision_dict[i], recall_dict[i], _ = precision_recall_curve(
                    y_test_binarized[:, i], y_score[:, i])
                ap_i = average_precision_score(y_test_binarized[:, i], y_score[:, i])
                average_prec.append(ap_i)

            # Macro-average
            # One naive way is to sample many points across recall and average. 
            # For demonstration, we’ll do a rough approach:
            mean_ap = np.mean(average_prec)

            # For plotting, let's pick an approach: 
            # we can combine all classes or plot the class that yields the macro average
            # We'll just plot the "micro" or "macro" curve. 
            # For simplicity, do a micro-average approach:
            y_true_micro = y_test_binarized.ravel()
            y_score_micro = y_score.ravel()
            precision_micro, recall_micro, _ = precision_recall_curve(y_true_micro, y_score_micro)
            ap_micro = average_precision_score(y_true_micro, y_score_micro)
            
            ax.plot(recall_micro, precision_micro,
                    label=f"{name} (macro-AP={mean_ap:.2f}, micro-AP={ap_micro:.2f})")
    else:
        # BINARY
        for model, name in zip(models, model_names):
            if not hasattr(model, "predict_proba"):
                logging.warning(f"Model {name} has no predict_proba; skipping PR curve.")
                continue

            y_score = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_score)
            ap = average_precision_score(y_test, y_score)
            ax.plot(recall, precision, label=f"{name} (AP={ap:.2f})")

    apply_common_grid(ax)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Merged Precision-Recall Curves")
    ax.legend(loc="best")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Merged Precision-Recall Curves saved to {save_path}")

    return fig


###############################################################################
# HELPER FOR MULTICLASS BINARIZATION
###############################################################################
def _binarize_labels(y_encoded, n_classes):
    """
    Converts an array of encoded labels (e.g., 0..4) 
    into a one-hot/binarized 2D array of shape [n_samples, n_classes].
    """
    binarized = np.zeros((len(y_encoded), n_classes))
    for i, label in enumerate(y_encoded):
        binarized[i, label] = 1
    return binarized

from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
# Make sure you have 'apply_common_grid' accessible
# from your current 'plots.py' code.

from sklearn.model_selection import learning_curve

def compute_learning_curve_data(model, X_train, y_train, cv=5, scoring="matthews_corrcoef"):
    """
    Computes learning curve data for a given model.

    Parameters:
        model: sklearn model (e.g., RandomForest, SVM, etc.)
        X_train: Training features
        y_train: Training labels
        cv: Number of cross-validation folds
        scoring: Metric to evaluate

    Returns:
        (train_sizes, train_mean, train_std, test_mean, test_std)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5)  # 5 points across training set size
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    return train_sizes, train_mean, train_std, test_mean, test_std

def plot_2x2_learning_curves(all_model_data, model_names, save_path=None):
    """
    Creates a single 2x2 figure of individual learning curves
    for 4 models. Each subplot calls the same plotting logic
    you currently have in 'plot_learning_curve'.

    Parameters
    ----------
    all_model_data : list of tuples
        Each tuple is (train_sizes, train_mean, train_std, test_mean, test_std).
        Must be length 4 if you want exactly 2x2 subplots.
    model_names : list of str
        Names for each model. Must match length of all_model_data.
    save_path : str or None
        If not None, the figure is saved to this path.

    Returns
    -------
    fig, axes : the Matplotlib Figure and Axes array.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Flatten axes array for easy iteration => [ax0, ax1, ax2, ax3]
    ax_list = axes.ravel()

    for i, (model_data, model_name) in enumerate(zip(all_model_data, model_names)):
        ax = ax_list[i]  # pick the subplot
        train_sizes, train_mean, train_std, test_mean, test_std = model_data

        # -- This is the same code as your 'plot_learning_curve' but using ax. --
        ax.plot(train_sizes, train_mean, color='blue', marker='o',
                markersize=5, label='Training MCC')
        ax.fill_between(train_sizes, train_mean + train_std,
                        train_mean - train_std, alpha=0.15, color='blue')
        ax.plot(train_sizes, test_mean, color='green', linestyle='--',
                marker='s', markersize=5, label='Validation MCC')
        ax.fill_between(train_sizes, test_mean + test_std,
                        test_mean - test_std, alpha=0.15, color='green')

        apply_common_grid(ax, which="both", linestyle="--", linewidth=0.7, color='gray')
        ax.set_xlabel('Number of training examples')
        ax.set_ylabel('MCC')
        ax.legend(loc='lower right')
        ax.set_ylim([-1.05, 1.05])
        ax.set_title(f'Learning Curve - {model_name}')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"2x2 Learning Curves figure saved to {save_path}")

    return fig, axes


def plot_2x2_roc_curves(models, model_names, X_test, y_test, label_encoder=None, save_path=None):
    """
    Creates a single 2x2 figure of individual ROC curves,
    one subplot per model. Handles multiclass using one-vs-rest (OvR).

    Parameters:
        models (list): List of trained models.
        model_names (list): List of model names.
        X_test (array): Test features.
        y_test (array): True labels (can be multiclass).
        label_encoder (LabelEncoder or None): If multiclass, used to binarize labels.
        save_path (str or None): Path to save the plot.

    Returns:
        fig, axes
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    import logging

    # Define the custom mapping for class labels.
    type_labels = {0: "Type I", 1: "Type II", 2: "Type III", 3: "Type IV", 4: "Unknown"}

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    # Identify unique classes for multiclass handling
    if label_encoder is not None and len(label_encoder.classes_) > 2:
        # Use integer-encoded classes
        classes = np.unique(y_test)
        n_classes = len(classes)
        y_test_bin = label_binarize(y_test, classes=classes)
        print(f"[DEBUG] y_test_bin shape: {y_test_bin.shape}")
    else:
        classes = np.unique(y_test)
        n_classes = len(classes)
        y_test_bin = None  # Not needed for binary classification

    for i, (model, model_name) in enumerate(zip(models, model_names)):
        ax = axes[i]

        if not hasattr(model, "predict_proba"):
            logging.warning(f"Model {model_name} has no predict_proba; skipping ROC curve.")
            ax.set_title(f"ROC Curve - {model_name}\nNo predict_proba()")
            ax.axis('off')
            continue

        if n_classes > 2:
            if y_test_bin is None:
                logging.error("y_test_bin is None for multiclass ROC curve.")
                ax.set_title(f"ROC Curve - {model_name}\nBinarization Failed")
                ax.axis('off')
                continue

            # Multiclass: One-vs-Rest strategy
            y_score = model.predict_proba(X_test)
            if y_score.shape[1] != n_classes:
                logging.error(f"Model {model_name} predict_proba output shape {y_score.shape} does not match number of classes {n_classes}.")
                ax.set_title(f"ROC Curve - {model_name}\nIncorrect predict_proba Shape")
                ax.axis('off')
                continue

            fpr_dict = {}
            tpr_dict = {}
            roc_auc_dict = {}

            for j in range(n_classes):
                fpr_dict[j], tpr_dict[j], _ = roc_curve(y_test_bin[:, j], y_score[:, j])
                roc_auc_dict[j] = auc(fpr_dict[j], tpr_dict[j])
                # Map the class label to the custom type label if possible
                try:
                    cls_int = int(classes[j])
                except ValueError:
                    cls_int = None
                label_str = type_labels.get(cls_int, f"Class {classes[j]}")
                ax.plot(fpr_dict[j], tpr_dict[j], lw=1,
                        label=f"{label_str} (AUC={roc_auc_dict[j]:0.2f})")

            ax.set_title(f'ROC Curve - {model_name}')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc="lower right")
            apply_common_grid(ax, which="both", linestyle="--", linewidth=0.7, color='gray')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])

            # Debugging Statements
            logging.debug(f"Model: {model_name}")
            logging.debug(f"FPR for classes: {fpr_dict}")
            logging.debug(f"TPR for classes: {tpr_dict}")
            logging.debug(f"AUC for classes: {roc_auc_dict}")

        else:
            # Binary classification
            y_score = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc_val = auc(fpr, tpr)

            ax.plot(fpr, tpr, lw=2, label=f'AUC={roc_auc_val:.2f}')
            ax.plot([0, 1], [0, 1], 'k--', label='Chance')
            ax.set_title(f'ROC Curve - {model_name}')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc="lower right")
            apply_common_grid(ax, which="both", linestyle="--", linewidth=0.7, color='gray')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])

            # Debugging Statements
            logging.debug(f"Model: {model_name}")
            logging.debug(f"FPR: {fpr}")
            logging.debug(f"TPR: {tpr}")
            logging.debug(f"AUC: {roc_auc_val}")

    # Handle any unused subplots (if number of models < 4)
    for j in range(i + 1, 4):
        axes[j].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"2x2 ROC Curves figure saved to {save_path}")

    plt.show()  # Ensure the plot is displayed
    return fig, axes

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score
import logging

def plot_2x2_precision_recall_curves(models, model_names, X_test, y_test, label_encoder=None, save_path=None):
    """
    Creates a single 2x2 figure of individual Precision-Recall curves,
    one subplot per model. Handles multiclass using one-vs-rest (OvR).

    Parameters:
        models (list): List of trained models.
        model_names (list): List of model names.
        X_test (array): Test features.
        y_test (array): True labels (can be multiclass).
        label_encoder (LabelEncoder or None): If multiclass, used to binarize labels.
        save_path (str or None): Path to save the plot.

    Returns:
        fig, axes
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    # Identify unique classes for multiclass handling
    if label_encoder is not None and len(label_encoder.classes_) > 2:
        classes = np.unique(y_test)
        n_classes = len(classes)
        y_test_bin = label_binarize(y_test, classes=classes)
    else:
        classes = np.unique(y_test)
        n_classes = len(classes)
        y_test_bin = None  # Not needed for binary classification

    for i, (model, model_name) in enumerate(zip(models, model_names)):
        ax = axes[i]
        
        if not hasattr(model, "predict_proba"):
            logging.warning(f"Model {model_name} has no predict_proba; skipping PR curve.")
            ax.set_title(f"PR Curve - {model_name}\nNo predict_proba()")
            ax.axis('off')
            continue

        if n_classes > 2:
            if y_test_bin is None:
                logging.error("y_test_bin is None for multiclass PR curve.")
                ax.set_title(f"PR Curve - {model_name}\nBinarization Failed")
                ax.axis('off')
                continue

            # Multiclass: One-vs-Rest strategy
            y_score = model.predict_proba(X_test)
            precision_dict = {}
            recall_dict = {}
            average_precision_dict = {}

            for j in range(n_classes):
                precision_dict[j], recall_dict[j], _ = precision_recall_curve(y_test_bin[:, j], y_score[:, j])
                average_precision_dict[j] = average_precision_score(y_test_bin[:, j], y_score[:, j])
                ax.plot(recall_dict[j], precision_dict[j], lw=1, label=f"Class {classes[j]} (AP={average_precision_dict[j]:0.2f})")

            ax.set_title(f'Precision-Recall Curve - {model_name}')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.legend(loc='lower left', fontsize='small')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.grid(which="both", linestyle="--", linewidth=0.7, color='gray')

        else:
            # Binary classification
            y_score = model.predict_proba(X_test)[:, 1]
            precision_val, recall_val, _ = precision_recall_curve(y_test, y_score)
            average_precision_val = average_precision_score(y_test, y_score)

            ax.plot(recall_val, precision_val, lw=2, label=f'AP={average_precision_val:.2f}')
            ax.plot([0, 1], [1, 0], 'k--', label='Chance')
            ax.set_title(f'Precision-Recall Curve - {model_name}')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.legend(loc='lower left')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.grid(which="both", linestyle="--", linewidth=0.7, color='gray')

    # Handle any unused subplots (if number of models < 4)
    for j in range(i + 1, 4):
        axes[j].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"2x2 Precision-Recall Curves figure saved to {save_path}")

    plt.show()
    return fig, axes

from sklearn.metrics import confusion_matrix

def compute_confusion_matrix(y_true, y_pred, all_labels):
    """
    Computes the confusion matrix.

    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        all_labels (list): List of all class labels.

    Returns:
        cm (array): Confusion matrix as a NumPy array.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(all_labels)))
    return cm

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import logging

def plot_2x2_confusion_matrices(confusion_matrices, model_names, all_labels, save_path=None, normalize=True, cmap='Blues'):
    """
    Plots four confusion matrices in a 2x2 layout with improved sizing, horizontal labels, and percentage values.
    
    Parameters:
        confusion_matrices (list): List of four confusion matrices (numpy arrays or pandas DataFrames).
        model_names (list): List of four model names.
        all_labels (list): List of class labels.
        save_path (str or None): Path to save the figure.
        normalize (bool): Whether to normalize the confusion matrices.
        cmap (str): Colormap for heatmap.
    
    Returns:
        fig, axes
    """
    if len(confusion_matrices) != 4 or len(model_names) != 4:
        logging.error("Exactly four confusion matrices and four model names are required for a 2x2 plot.")
        raise ValueError("Provide exactly four confusion matrices and four corresponding model names.")
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # Reduced figure size
    axes = axes.flatten()

    for i, (cm, model_name) in enumerate(zip(confusion_matrices, model_names)):
        ax = axes[i]
        
        # If cm is a DataFrame, ensure the order of labels
        if isinstance(cm, pd.DataFrame):
            cm = cm.reindex(index=all_labels, columns=all_labels)
            cm = cm.fillna(0).values
        else:
            # Assume cm is a NumPy array; ensure it's in the correct order
            cm = np.array(cm)
            if cm.shape[0] != len(all_labels) or cm.shape[1] != len(all_labels):
                logging.error(f"Confusion matrix shape {cm.shape} does not match number of labels {len(all_labels)}.")
                raise ValueError(f"Confusion matrix shape {cm.shape} does not match number of labels {len(all_labels)}.")
        
        # Normalize to percentage
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
        cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
        annotations = np.array([[f'{val:.2f}%' for val in row] for row in cm_normalized])
        
        sns.heatmap(cm_normalized, annot=annotations, fmt='', cmap=cmap, ax=ax, cbar=False,
                    xticklabels=all_labels, yticklabels=all_labels, annot_kws={"size": 8}, square=True)
        ax.set_title(f'{model_name}', fontsize=10, fontweight='bold')

        ax.set_xlabel('Predicted Label', fontsize=8)
        ax.set_ylabel('True Label', fontsize=8)
        ax.tick_params(axis='x', rotation=0, labelsize=8)  # Set horizontal labels
        ax.tick_params(axis='y', rotation=0, labelsize=8)
        ax.grid(False)  # Removed excessive grid styling

    # Hide any unused subplots if more than four
    for j in range(4, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"2x2 Confusion Matrices figure saved to {save_path}")

    plt.show()
    return fig, axes


def generate_class_metrics_table(y_true, y_pred, all_labels):
    """
    Generates a table with MCC, Accuracy, Precision, Recall, and F1-Score for each class.
    
    Parameters:
    -----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    all_labels : list of str
        List of all class labels.
    
    Returns:
    --------
    metrics_table : pandas.DataFrame
        DataFrame containing the metrics per class.
    """
    from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score

    metrics = {
        'Class': [],
        'MCC': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': []
    }

    for idx, label in enumerate(all_labels):
        # One-vs-Rest approach
        y_true_binary = (y_true == idx).astype(int)
        y_pred_binary = (y_pred == idx).astype(int)
        
        # Compute Confusion Matrix
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        TN, FP, FN, TP = cm.ravel()
        
        # Compute Metrics
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        denominator = np.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))
        mcc = ((TP * TN) - (FP * FN)) / denominator if denominator != 0 else 0

        # Append to metrics dictionary
        metrics['Class'].append(label)
        metrics['MCC'].append(round(mcc, 4))
        metrics['Accuracy'].append(round(accuracy, 4))
        metrics['Precision'].append(round(precision, 4))
        metrics['Recall'].append(round(recall, 4))
        metrics['F1-Score'].append(round(f1, 4))
    
    # Create DataFrame
    metrics_table = pd.DataFrame(metrics)
    
    return metrics_table

