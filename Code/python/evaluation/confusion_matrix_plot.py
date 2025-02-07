import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config.config import PLOT_SAVE_DIR
from visualization.plots import save_plot


def compute_confusion_matrix(model, X_test, y_test, labels, dataset_name, predictor_type, model_name, label_encoder=None):
    """
    Compute and plot the confusion matrix for a given model.
    """
    y_pred = model.predict(X_test)

    # Decode labels if needed
    y_test_decoded, y_pred_decoded = decode_labels(y_test, y_pred, label_encoder)

    confmat = confusion_matrix(y_test_decoded, y_pred_decoded, labels=labels)
    logging.info(f"Confusion Matrix for {model_name}:")
    logging.info(confmat)

    fig = plot_confusion_matrix(confmat, labels, model_name)
    plot_filename = f'confusion_matrix_{dataset_name}_{predictor_type}_{model_name}.png'
    save_plot(fig, plot_filename)
    

def decode_labels(y_test, y_pred, label_encoder):
    """
    Decode labels if a label encoder is provided.

    Args:
        y_test (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        label_encoder (LabelEncoder): Label encoder instance.

    Returns:
        tuple: Decoded true and predicted labels.
    """
    if label_encoder is not None:
        y_test_decoded = label_encoder.inverse_transform(y_test)
        if y_pred.dtype.kind in ['i', 'u']:  # integer/unsigned integer
            y_pred_decoded = label_encoder.inverse_transform(y_pred)
        else:
            y_pred_decoded = y_pred
    else:
        y_test_decoded = y_test
        y_pred_decoded = y_pred

    return y_test_decoded, y_pred_decoded
