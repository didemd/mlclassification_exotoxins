import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    matthews_corrcoef,
    accuracy_score,
    precision_recall_fscore_support
)
from numpy.random import choice, seed
from numpy import mean, std
from tabulate import tabulate
from sklearn.preprocessing import LabelEncoder

def bootstrap_metric(metric, y_true, y_pred, n_boot):
    """
    Perform bootstrapping to calculate the average metric and its standard error.
    """
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)
    metric_accumulated = []
    seed(7)

    for _ in range(n_boot):
        try:
            bootstrap_sample_index = choice(y_true.index, size=len(y_true), replace=True)
            bootstrap_sample_ytrue = y_true[bootstrap_sample_index]
            bootstrap_sample_ypred = y_pred[bootstrap_sample_index]
            boot_metric = metric(bootstrap_sample_ytrue, bootstrap_sample_ypred)
            metric_accumulated.append(boot_metric)
        except Exception as e:
            logging.error(f"Error during bootstrapping: {e}")
            metric_accumulated.append(0.0)

    try:
        avg_metric = round(mean(metric_accumulated), 4)
        metric_se = round(std(metric_accumulated, ddof=1), 4) * 1.96
    except Exception as e:
        logging.error(f"Error calculating mean and SE: {e}")
        avg_metric, metric_se = 0.0, 0.0

    return avg_metric, metric_se


def calculate_evaluation_metrics(true_labels, predicted_labels, label_encoder=None, n_boot=1000, average_method='macro'):
    """
    Calculate evaluation metrics (MCC, Accuracy, Precision, Recall, F1-Score) 
    and return each as (mean, standard error).
    """
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    #check if labels are numeric
    labels_are_numeric = np.issubdtype(true_labels.dtype, np.integer) and np.issubdtype(predicted_labels.dtype, np.integer)

    
    if label_encoder is not None and not labels_are_numeric:
        # Only transform if labels are strings
        try:
            true_labels = label_encoder.transform(true_labels)
            predicted_labels = label_encoder.transform(predicted_labels)
        except ValueError as ve:
            logging.error(f"LabelEncoder failed to transform labels: {ve}")
            return {}
    elif label_encoder is not None and labels_are_numeric:
        # Assume labels are already encoded; ensure label_encoder classes include all
        max_label = max(true_labels.max(), predicted_labels.max())
        if max_label >= len(label_encoder.classes_):
            logging.error(f"LabelEncoder classes do not cover label '{max_label}'.")
            return {}
    elif label_encoder is None and not labels_are_numeric:
        # No encoder provided, but labels are strings; fit a new encoder
        label_encoder = LabelEncoder()
        true_labels = label_encoder.fit_transform(true_labels)
        predicted_labels = label_encoder.transform(predicted_labels)

    # Auto-detect binary/multiclass
    unique_labels = np.unique(true_labels)
    if len(unique_labels) > 2:
        average_method = 'macro'
        logging.info(f"Detected multiclass problem. Using average='{average_method}'.")

    metrics_with_error = {}

    try:
        # 1. MCC
        mcc_val, mcc_se = bootstrap_metric(matthews_corrcoef, true_labels, predicted_labels, n_boot)
        metrics_with_error['MCC'] = (mcc_val, mcc_se)

        # 2. Accuracy
        acc_val, acc_se = bootstrap_metric(accuracy_score, true_labels, predicted_labels, n_boot)
        metrics_with_error['Accuracy'] = (acc_val, acc_se)

        # 3. Precision
        prec_val, prec_se = bootstrap_metric(
            lambda y, y_pred: precision_recall_fscore_support(y, y_pred, average=average_method)[0],
            true_labels,
            predicted_labels,
            n_boot
        )
        metrics_with_error['Precision'] = (prec_val, prec_se)

        # 4. Recall
        recall_val, recall_se = bootstrap_metric(
            lambda y, y_pred: precision_recall_fscore_support(y, y_pred, average=average_method)[1],
            true_labels,
            predicted_labels,
            n_boot
        )
        metrics_with_error['Recall'] = (recall_val, recall_se)

        # 5. F1-Score
        f1_val, f1_se = bootstrap_metric(
            lambda y, y_pred: precision_recall_fscore_support(y, y_pred, average=average_method)[2],
            true_labels,
            predicted_labels,
            n_boot
        )
        metrics_with_error['F1-Score'] = (f1_val, f1_se)

    except Exception as e:
        logging.error(f"Error calculating metrics: {e}")

    # Validate all metrics are tuples of length 2
    for key, value in metrics_with_error.items():
        if not isinstance(value, tuple) or len(value) != 2:
            logging.error(f"Invalid structure for metric '{key}': {value}")
            metrics_with_error[key] = (0.0, 0.0)

    return metrics_with_error


def generate_table(metrics_with_errors):
    """
    Generate a table in the desired format with metrics and their corresponding errors.

    Args:
        metrics_with_errors (dict): 
          e.g. {'MCC': (0.56, 0.12), 'Accuracy': (0.84, 0.05), ... }

    Returns:
        str: Table formatted with tabulate, printing "error ± value".
    """
    table_data = []

    for key, value in metrics_with_errors.items():
        if isinstance(value, (list, tuple)) and len(value) == 2:
            metric_value, error_value = value
            table_data.append(f"{error_value:.3f} ± {metric_value:.3f}")
        else:
            logging.error(f"Invalid structure for metric '{key}': {value}")
            table_data.append("N/A")

    headers = list(metrics_with_errors.keys())
    return tabulate([table_data], headers=headers, tablefmt="grid")

def save_metrics_to_csv(metrics_with_errors, file_name="metrics_with_errors.csv"):
    data = {}
    for metric, value in metrics_with_errors.items():
        if isinstance(value, (list, tuple)) and len(value) == 2:
            mean_val, error_val = value
            data[metric] = [f"{mean_val:.3f} ± {error_val:.3f}"]
        else:
            logging.error(f"Invalid structure for metric '{metric}': {value}")
            data[metric] = ["N/A"]

    pd.DataFrame(data).to_csv(file_name, index=False)
    print(f"Metrics saved to {file_name}")

def save_table_to_file(table, file_name="metrics_table.txt"):
    with open(file_name, "w") as f:
        f.write(table)
    print(f"Table saved to {file_name}")

# Supporting helper functions
def decode_labels(true_labels, predicted_labels, label_encoder):
    try:
        if (np.issubdtype(true_labels.dtype, np.integer) and 
            isinstance(label_encoder, LabelEncoder)):
            true_labels = label_encoder.inverse_transform(true_labels)
            if np.issubdtype(predicted_labels.dtype, np.integer):
                predicted_labels = label_encoder.inverse_transform(predicted_labels)
    except ValueError as e:
        logging.error(f"Error decoding labels: {e}")
        raise
    return true_labels, predicted_labels

def log_unique_labels(true_labels, predicted_labels, label_type):
    unique_true = np.unique(true_labels)
    unique_pred = np.unique(predicted_labels)
    logging.info(f"Unique true labels for {label_type}: {', '.join(map(str, unique_true))}")
    logging.info(f"Unique predicted labels for {label_type}: {', '.join(map(str, unique_pred))}")

def ensure_binary_classification(true_labels):
    unique_labels = np.unique(true_labels)
    if len(unique_labels) != 2:
        raise ValueError(f"Expected 2 unique labels, but got {len(unique_labels)}: {unique_labels}")

def determine_positive_label(true_labels, label_type, preferred_positive='Vertebrate'):
    unique_labels = np.unique(true_labels)
    pos_label = preferred_positive if preferred_positive in unique_labels else unique_labels[0]
    return pos_label
