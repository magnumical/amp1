from sklearn.metrics import classification_report
import mlflow
    
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
 
import numpy as np


def log_metrics(y_true, y_pred, mode):
    """Log evaluation metrics."""
    precision = classification_report(y_true, y_pred, output_dict=True)['weighted avg']['precision']
    recall = classification_report(y_true, y_pred, output_dict=True)['weighted avg']['recall']
    f1_score = classification_report(y_true, y_pred, output_dict=True)['weighted avg']['f1-score']

    mlflow.log_metric(f"{mode}_precision", precision)
    mlflow.log_metric(f"{mode}_recall", recall)
    mlflow.log_metric(f"{mode}_f1_score", f1_score)




def plot_roc_curve(y_true, y_pred_prob, mode, class_names=None):
    """
    Plot ROC curve for binary or multi-class classification.

    Args:
        y_true: True labels (array-like).
        y_pred_prob: Predicted probabilities (array-like).
        mode: Mode of classification ('binary' or 'multi-class').
        class_names: List of class names (optional, required for multi-class).
    """
    plt.figure(figsize=(10, 7))

    if mode == 'binary':
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob[:, 1])
        auc_score = roc_auc_score(y_true, y_pred_prob[:, 1])
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (area = {auc_score:.2f})")
    else:
        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_true == i, y_pred_prob[:, i])
            auc_score = roc_auc_score(y_true == i, y_pred_prob[:, i])
            plt.plot(fpr, tpr, lw=2, label=f"Class {class_name} (AUC = {auc_score:.2f})")

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({mode})")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()



import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, class_names, mode):
    """
    Plot confusion matrix for binary or multi-class classification.

    Args:
        y_true: True labels (array-like).
        y_pred: Predicted labels (array-like).
        class_names: List of class names.
        mode: Mode of classification ('binary' or 'multi-class').
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix ({mode})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
