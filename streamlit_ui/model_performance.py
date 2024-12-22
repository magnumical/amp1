import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import numpy as np
import os

# Paths and Constants
MODEL_PATH = "./models"
DATASET_PATH = "./processed_datasets"
SUMMARY_FILE = "./streamlit_ui/model_evaluation_summary.csv"
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

# Function to evaluate models
def evaluate_models():
    metrics_dict = []
    for model_name in MODELS:
        mode_key = model_name.replace("final_model_", "").replace(".h5", "").replace(" ", "_").lower()
        dataset = DATASETS.get(mode_key)

        if dataset:
            model_path = os.path.join(MODEL_PATH, model_name)
            model = load_model(model_path)

            X_test_path, y_test_path = dataset
            X_test = np.load(os.path.join(DATASET_PATH, X_test_path))
            y_test = np.load(os.path.join(DATASET_PATH, y_test_path))

            y_pred_prob = model.predict(X_test)
            y_pred = np.argmax(y_pred_prob, axis=1)
            y_true = np.argmax(y_test, axis=1)

            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')

            metrics_dict.append({
                "Model": mode_key,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "ROC-AUC": auc
            })

    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv(SUMMARY_FILE, index=False)
    return metrics_df

# Streamlit App Function
def run():
    st.header("Model Performance")

    # Check if the summary file exists
    if os.path.exists(SUMMARY_FILE):
        st.info("Loading precomputed metrics from summary file.")
        metrics_df = pd.read_csv(SUMMARY_FILE)
    else:
        # Add a loading spinner while the evaluation is running
        with st.spinner("Processing models and generating metrics, please wait..."):
            metrics_df = evaluate_models()
        st.success("Model evaluation completed and saved!")

    st.subheader("Model Metrics")
    st.dataframe(metrics_df)

    st.subheader("Metrics Visualization")
    if not metrics_df.empty:
        metric_to_plot = st.selectbox("Select Metric to Visualize", metrics_df.columns[1:])
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=metrics_df, x="Model", y=metric_to_plot, ax=ax, palette="viridis",legend=False)
        ax.set_title(f"Model {metric_to_plot}", fontsize=16, fontweight='bold')
        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel(metric_to_plot, fontsize=12)
        st.pyplot(fig)

    # Additional Image Types
    st.subheader("Additional Visualizations")
    st.text("Below are other types of visualizations for analysis:")

    # Heatmap of Metrics
    st.subheader("Metrics Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(metrics_df.set_index("Model"), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Heatmap of Model Metrics", fontsize=16, fontweight='bold')
    st.pyplot(fig)

    # Pairplot of Metrics
    st.subheader("Metrics Pairplot")
    if len(metrics_df) > 1:  # Pairplot requires at least two rows
        fig = sns.pairplot(metrics_df.drop("Model", axis=1))
        st.pyplot(fig=fig.fig)
