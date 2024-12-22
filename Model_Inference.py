

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Paths
MODEL_PATH = "./models"
DATASET_PATH = "./processed_datasets"

# Model and dataset filenames
MODELS = [
    "final_model_binary_augmented.h5",
    "final_model_binary_log_mel.h5",
    "final_model_binary_mfcc.h5",
    "final_model_multi_augmented.h5",
    "final_model_multi_log_mel.h5",
    "final_model_multi_mfcc.h5"
]

DATASETS = {
    "binary_augmented": ("X_test_binary_augmented.npy", "y_test_binary_augmented.npy"),
    "binary_log_mel": ("X_test_binary_log_mel.npy", "y_test_binary_log_mel.npy"),
    "binary_mfcc": ("X_test_binary_mfcc.npy", "y_test_binary_mfcc.npy"),
    "multi_augmented": ("X_test_multi_augmented.npy", "y_test_multi_augmented.npy"),
    "multi_log_mel": ("X_test_multi_log_mel.npy", "y_test_multi_log_mel.npy"),
    "multi_mfcc": ("X_test_multi_mfcc.npy", "y_test_multi_mfcc.npy")
}

# Metrics dictionary
metrics_dict = []

# Function to evaluate a model
def evaluate_model(model, X_test, y_test, mode):
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    print(f"--- Evaluation for {mode} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\n")

    # Log metrics
    metrics_dict.append({
        "Model": mode,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": auc
    })

    # Plot ROC curve
    fpr = {}
    tpr = {}
    for i in range(y_test.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_prob[:, i])
    plt.figure(figsize=(10, 6))
    for i, label in enumerate(np.unique(y_true)):
        plt.plot(fpr[i], tpr[i], label=f"Class {label} ROC")
    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC Curve - {mode}")
    plt.legend()
    plt.savefig(f"roc_curve_{mode}.png")
    plt.close()

# Evaluate all models
for model_name in MODELS:
    mode_key = model_name.replace("final_model_", "").replace(".h5", "").replace(" ", "_").lower()
    dataset = DATASETS.get(mode_key)

    if dataset:
        # Load the model and dataset
        model_path = os.path.join(MODEL_PATH, model_name)
        model = load_model(model_path)

        X_test_path, y_test_path = dataset
        X_test = np.load(os.path.join(DATASET_PATH, X_test_path))
        y_test = np.load(os.path.join(DATASET_PATH, y_test_path))

        # Evaluate the model
        evaluate_model(model, X_test, y_test, mode_key)
    else:
        print(f"No dataset found for model: {model_name}")

# Save metrics as a CSV
metrics_df = pd.DataFrame(metrics_dict)
metrics_df.to_csv("model_evaluation_summary.csv", index=False)
print("Evaluation complete. Summary saved as 'model_evaluation_summary.csv'.")
