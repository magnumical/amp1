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
    "binary_log_mel": ("X_test_binary_log_mel.npy", "y_test_binary_log_mel.nlidpy"),
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
            if not os.path.exists(model_path):
                st.error(f"Model file not found: {model_path}")
                continue
            
            try:
                model = load_model(model_path)
            except Exception as e:
                st.error(f"Error loading model {model_name}: {e}")
                continue

            X_test_path, y_test_path = dataset
            try:
                X_test = np.load(os.path.join(DATASET_PATH, X_test_path))
                y_test = np.load(os.path.join(DATASET_PATH, y_test_path))
            except Exception as e:
                st.error(f"Error loading dataset for {mode_key}: {e}")
                continue

            # Reshape X_test if needed
            if len(X_test.shape) == 2 and model.input_shape[1:] == (X_test.shape[1], 1):
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            if X_test.shape[1:] != model.input_shape[1:]:
                st.error(f"Shape mismatch: Model {model_name} expects {model.input_shape[1:]}, but dataset has {X_test.shape[1:]}")
                continue

            try:
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
            except Exception as e:
                st.error(f"Error during evaluation for model {model_name}: {e}")
                continue

    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv(SUMMARY_FILE, index=False)
    return metrics_df

# Streamlit App Function
def run():
    st.header("Model Performance")

    st.subheader("Model Training Scheme")
    st.write(""" Below is the illusteration of the model training and evaluation framework. The framework includes the following steps:
    """)

    st.image("./streamlit_ui/img/training.png", caption="Model Training and Evaluation Framework")

    st.subheader("Types of Models and Input Features")
    st.write("""The workflow supports six distinct models, combining two classification settings (binary and multi-class) with three feature extraction modes (MFCC, Log-Mel Spectrogram, and Augmented Features). 
    Binary classification models distinguish between Normal and Abnormal, while multi-class models categorize audio into broader conditions like Normal, Chronic Respiratory Diseases, and Respiratory Infections.
    MFCC and Log-Mel features require 2D CNN architectures due to their structured, matrix-like representations, whereas augmented features leverage 1D CNNs to handle diverse feature sets. 
    """)


    st.subheader("Feature Types and Their Characteristics")

    st.markdown("""
    | **Feature Mode** | **Definition**                                                                                             | **Representation Type** | **Model Type Required** |
    |-------------------|-----------------------------------------------------------------------------------------------------------|--------------------------|--------------------------|
    | **MFCC**         | Represents short-term power spectrum using the Mel scale, emphasizing human auditory perception.           | 2D Matrix               | 2D CNN                  |
    | **Log-Mel**      | A spectrogram emphasizing quieter sounds by applying a logarithmic scale to Mel frequencies.               | 2D Matrix               | 2D CNN                  |
    | **Augmented**    | Features derived from MFCCs with added variability through augmentation techniques such as noise addition, pitch shifting, and time stretching. | 1D Sequence             | 1D CNN                  |
    """)

    st.write("""
    Using multiple feature types allows models to leverage diverse characteristics of the audio data, ensuring better generalization and robustness:
    - **MFCC:** Captures the most relevant features of human auditory perception.
    - **Log-Mel:** Highlights quieter components, providing a balanced representation of both loud and soft sounds.
    - **Augmented:** Adds variability to training data, improving robustness and reducing overfitting risks.
    """)





    st.subheader("Training, Optimization, and Evaluation")

    st.markdown("""
    ### **Model Building Process**
    The model building process was built to handle both binary and multi-class classification.  
    The architecture is designed dynamically to adapt to the input features and classification requirements. 

    - **Binary Classification:** The output layer uses a single neuron with a sigmoid activation function.
    - **Multi-Class Classification:** The output layer uses softmax activation with a number of neurons equal to the classes. 


    ---

    ### **Why Both Binary and Multi-Class?**
    My goal was creating a `Comprehensive Testing Framework` that we could test different types of input, models, and outputs to see which one suits well!
    This is usually what I do in my projects, so the overall idea is creating a framework that you can subtitute its compoionents. For example, you can implement **Audio Spectrogram Transformer** and add it to the testing framework easily.
    - **Binary Classification:** Focuses on distinguishing Normal from Abnormal cases.  
    This approach simplifies the task --> but it faces overfitting and generalization issues.

    - **Multi-Class Classification:** Groups Abnormal cases into specific categories (e.g., Chronic Respiratory Diseases, Respiratory Infections).  
    Due to data imbalance, some groups with fewer than five cases were merged together. I searched about disease and grouped them based on some papers that I read. This granularity was between binary and all-class classification.

    ---

    ### **Training Scheme**
    Models are trained with:
    - **Loss Functions:**
    - `binary_crossentropy` for binary classification.
    - `categorical_crossentropy` for multi-class classification.
    - **Regularization:** Dropout layers mitigate overfitting.
    - **Optimizer:** Adamax is chosen for its adaptability to varying learning rates.
    - **Early Stopping:** Monitors validation loss and halts training if no improvement is observed after five consecutive epochs.

    ---

    ### **Hyperparameter Optimization**
    Optimization with Optuna tunes:
    - Number of filters in convolutional layers.
    - Units in dense layers.
    - Dropout rates.
    - Learning rates.

    ---

    ### **Evaluation Landscape**
    Models are evaluated using:
    - **Validation Accuracy:** Ensures performance during training.
    - **Test Metrics:** Precision, recall, F1-score, confusion matrix, and ROC-AUC.

    ---

    ### **Model Selection**
    The best model for each feature mode and classification task is selected based on:
    - Performance on unseen test data.
    - Robustness to imbalances and augmentation artifacts.

    ---

    ### **Challenges Addressed**
    - Balancing complexity (multi-class) with generalizability (binary) --> sice data is highly imbalanced!
    - Handling diverse feature representations and their impact on model performance.
    - Making the framework modifiable for future.
    """)

    st.subheader("Model Names and Configurations")

    st.markdown("""
    To keep the visualizations concise, the models are referred to using short names in the images. Below is the mapping between these names and their configurations:

    ### **Binary Classification Models**
    - **binary_augmented:** Uses augmented features.  
    Model file: `final_model_binary_augmented.h5`
    - **binary_log_mel:** Uses Log-Mel Spectrogram features.  
    Model file: `final_model_binary_log_mel.h5`
    - **binary_mfcc:** Uses MFCC features.  
    Model file: `final_model_binary_mfcc.h5`

    Classes:  
    - **Abnormal**
    - **Normal**

    ---

    ### **Multi-Class Classification Models**
    - **multi_augmented:** Uses augmented features.  
    Model file: `final_model_multi_augmented.h5`
    - **multi_log_mel:** Uses Log-Mel Spectrogram features.  
    Model file: `final_model_multi_log_mel.h5`
    - **multi_mfcc:** Uses MFCC features.  
    Model file: `final_model_multi_mfcc.h5`

    Classes:  
    - **Chronic Respiratory Diseases**
    - **Normal**
    - **Respiratory Infections**
    """)



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
