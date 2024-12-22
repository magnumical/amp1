from sklearn.metrics import classification_report
import mlflow

 
import numpy as np

def log_metrics(y_true, y_pred, mode):
    """
    Log evaluation metrics for binary or multi-class classification.

    Args:
        y_true: True labels (array-like, one-hot encoded for multi-class).
        y_pred: Predicted probabilities (array-like, continuous values).
        mode: Mode of classification ('binary' or 'multi-class').
    """
    # Convert one-hot encoded `y_true` to class indices
    if y_true.ndim > 1:  # If one-hot encoded
        y_true = np.argmax(y_true, axis=1)
    
    # Convert predicted probabilities `y_pred` to class indices
    if y_pred.ndim > 1:  # If predicted as probabilities
        y_pred = np.argmax(y_pred, axis=1)

    if mode == 'binary':
        class_names = ["Class 0", "Class 1"]
        classification = classification_report(y_true, y_pred, output_dict=True, target_names=class_names)
    else:
        unique_classes = np.unique(y_true)
        class_names = [f"Class {i}" for i in unique_classes]
        classification = classification_report(y_true, y_pred, output_dict=True, target_names=class_names)

    # Log metrics to MLflow
    precision = classification['weighted avg']['precision']
    recall = classification['weighted avg']['recall']
    f1_score = classification['weighted avg']['f1-score']

    mlflow.log_metric(f"{mode}_precision", precision)
    mlflow.log_metric(f"{mode}_recall", recall)
    mlflow.log_metric(f"{mode}_f1_score", f1_score)

    print(f"Classification Report ({mode}):\n", classification_report(y_true, y_pred, target_names=class_names))

    
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np

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
