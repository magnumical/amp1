import logging
import numpy as np
from utils.data_loader import load_data, process_audio_metadata
from utils.audioprocessing import *
from utils.model_utils import * 
import joblib
from utils.evaluation import log_metrics, plot_roc_curve, plot_confusion_matrix
import os
import gc
from joblib import Parallel, delayed
import mlflow
import mlflow.keras
import pandas as pd
import librosa
import librosa.display
import optuna
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical, normalize
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, GRU, Input, add, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs from TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
data_logger = logging.getLogger("data_pipeline")
train_logger = logging.getLogger("train")
processing_logger = logging.getLogger("data_processing")
model_logger = logging.getLogger("model_training")

# Dataset and Paths
AUDIO_FILES_PATH = './/data//Respiratory_Sound_Database//audio_and_txt_files'

def save_dataset(X, y, mode, output_dir="./processed_datasets/new"):
    """
    Save the processed X and y to .npy files.
    """
    import os
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save files with mode-specific names
    X_path = os.path.join(output_dir, f"X_{mode}.npy")
    y_path = os.path.join(output_dir, f"y_{mode}.npy")
    
    np.save(X_path, X)
    np.save(y_path, y)
    processing_logger.info(f"Saved dataset for mode '{mode}' to {output_dir}")


def load_or_process_dataset(df_filtered, audio_files_path, mode, feature_type, output_dir="processed_datasets/new"):

    # File paths for preprocessed data
    X_path = os.path.join(output_dir, f"X_{mode}.npy")
    y_path = os.path.join(output_dir, f"y_{mode}.npy")

    # Check if the files exist
    if os.path.exists(X_path) and os.path.exists(y_path):
        processing_logger.info(f"Preprocessed files found for mode '{mode}'. Loading from disk...")
        X = np.load(X_path)
        y = np.load(y_path)
    else:
        processing_logger.info(f"Preprocessed files not found for mode '{mode}'. Processing data...")
        os.makedirs(output_dir, exist_ok=True)

        # Prepare the dataset
        if feature_type == 'augmented':
            X, y, le = prepare_dataset_augmented(
                df_filtered, 
                audio_files_path,
                classification_mode=mode
            )
        else:
            X, y, le = prepare_dataset_parallel(
                df_filtered, 
                audio_files_path,
                mode=feature_type,
                classification_mode=mode
            )
            
        # Save the processed data and LabelEncoder
        np.save(X_path, X)
        np.save(y_path, y)
        processing_logger.info(f"Saved processed dataset and LabelEncoder for mode '{mode}' to {output_dir}")

    le = LabelEncoder()
    return X, y, le

def log_class_distribution(y, message):
    """Log the class distribution."""
    if y.ndim == 1:  # Binary classification (1D array of 0s and 1s)
        unique, counts = np.unique(y, return_counts=True)
    else:  # Multi-class classification (2D one-hot encoded array)
        unique, counts = np.unique(np.argmax(y, axis=1), return_counts=True)

    class_distribution = dict(zip(unique, counts))
    processing_logger.info(f"{message} Class Distribution: {class_distribution}")


def generate_random_audio_data(samples=200, feature_dim=20):
    """Generate random audio-like data for testing purposes."""
    X = np.random.rand(samples, feature_dim, feature_dim)  # Simulate 2D audio features
    y = np.random.randint(0, 2, size=samples)  # Binary classification labels
    return X, y

def test_model():
    """Test 2D CNN model with simulated audio data for debugging."""
    print("[DEBUG] Generating simulated audio data...")
    global X_train, X_val, X_test, y_train, y_val, y_test
    X, y = generate_random_audio_data()

    # Simulate preprocessing similar to audio processing pipeline
    print("[DEBUG] Preprocessing simulated audio data...")
    X_preprocessed = np.array([np.log1p(sample) for sample in X])  # Simulate a log transform or feature extraction

    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X_preprocessed, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    print(f"[DEBUG] Data split: Training={X_train.shape}, Validation={X_val.shape}, Test={X_test.shape}")

    # Expand dimensions for 2D CNN input
    X_train = np.expand_dims(X_train, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    print("[DEBUG] Initializing 2D CNN model...")
    model = track_experiment_with_mlflow_and_optuna(
        mode='mfcc',
        num_classes=1,
        model_type='2D',  # Specify 2D CNN for MFCC and Log-Mel
        classification_mode='binary',
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_trials=20,         
    )

    print("[DEBUG] Training the model...")
    # Train the model with a single epoch for testing
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=32)

    print("[DEBUG] Evaluating the model...")
    results = model.evaluate(X_test, y_test)
    print(f"[DEBUG] Test evaluation results: {results}")


# Define main function
def main():
    # python Train.py --metadata_path data/Respiratory_Sound_Database/audio_and_txt_files --audio_files_path data/Respiratory_Sound_Database/audio_and_txt_filesv --demographic_path data/demographic_info.tx --diagnosis_path --diagnosis_path data/Respiratory_Sound_Database/patient_diagnosis.csv --classification_modes binary --feature_types mfcc 

    parser = argparse.ArgumentParser(description="Run the respiratory sound analysis pipeline.")
    parser.add_argument("--metadata_path", type=str, default="./data/metadata", help="Path to the metadata directory.")
    parser.add_argument("--audio_files_path", type=str, default="./data/audio", help="Path to the directory containing audio files.")
    parser.add_argument("--demographic_path", type=str, default="./data/demographic_info.txt", help="Path to the demographic info file.")
    parser.add_argument("--diagnosis_path", type=str, default="./data/patient_diagnosis.csv", help="Path to the patient diagnosis CSV file.")
    parser.add_argument("--tracking_uri", type=str, default="./mlruns", help="MLflow tracking URI.")
    parser.add_argument("--classification_modes", type=str, nargs='+', default=['multi', 'binary'], help="Classification modes to run. Options: 'binary', 'multi'.")
    parser.add_argument("--feature_types", type=str, nargs='+', default=['mfcc'], help="Feature types to use. Options: 'mfcc', 'log_mel', 'augmented'.")
    parser.add_argument("--debug", action='store_true', help="Run in debug mode with random test data.")
    args = parser.parse_args()


    if args.debug:
        test_model()
        return

    # Set up directories and MLflow tracking
    AUDIO_FILES_PATH = args.audio_files_path
    mlflow.set_tracking_uri(args.tracking_uri)

    # Logging initial information
    data_logger.info("Starting data pipeline.")
    data_logger.info("Loading and preprocessing data...")

    # Load and preprocess data
    df = load_data(
        diagnosis_path=args.diagnosis_path,
        demographic_path=args.demographic_path
    )
    audio_metadata = process_audio_metadata(AUDIO_FILES_PATH)
    df_all = merge_datasets(audio_metadata, df)

    models = []

    for classification_mode in args.classification_modes:
        # Preprocess dataset for classification mode
        df_filtered = filter_and_sample_data(df_all, mode=classification_mode)
        processing_logger.info(f"Dataset shape for {classification_mode} mode: {df_filtered.shape}")

        for feature_type in args.feature_types:
            processing_logger.info(f"Running experiment for {classification_mode} classification with {feature_type} features.")
            
            # Load or process dataset
            X, y, le = load_or_process_dataset(
                df_filtered, AUDIO_FILES_PATH,
                feature_type=feature_type,
                mode=classification_mode,
                output_dir=f"processed_datasets/{classification_mode}"
            )

            # Split data into train/val/test
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

            # Save test data for future evaluation
            np.save(f"X_test_{classification_mode}_{feature_type}.npy", X_test)
            np.save(f"y_test_{classification_mode}_{feature_type}.npy", y_test)
            mlflow.log_artifact(f"X_test_{classification_mode}_{feature_type}.npy")
            mlflow.log_artifact(f"y_test_{classification_mode}_{feature_type}.npy")

            # Log dataset characteristics
            log_class_distribution(y_train, "Before Oversampling")
            processing_logger.info(f"Train size: {X_train.shape}, Validation size: {X_val.shape}, Test size: {X_test.shape}")

            try:
                X_train, y_train = oversample_data(X_train, y_train)
            except ValueError as e:
                processing_logger.warning(f"SMOTE skipped: {e}")
            log_class_distribution(y_train, "After Oversampling")

            # Determine number of classes
            num_classes = 1 if classification_mode == "binary" else y_train.shape[1]

            # Train and save model
            with mlflow.start_run(run_name=f"Experiment_{classification_mode}_{feature_type}", nested=True):
                if feature_type == 'augmented':
                    X_train = np.expand_dims(X_train, axis=-1)
                    X_val = np.expand_dims(X_val, axis=-1)
                    X_test = np.expand_dims(X_test, axis=-1)

                    model = track_experiment_with_mlflow_and_optuna(
                        mode=feature_type,
                        num_classes=num_classes,
                        model_type='1D',
                        classification_mode=classification_mode,
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        n_trials=20,
                    )
                else:
                    model = track_experiment_with_mlflow_and_optuna(
                        mode=feature_type,
                        num_classes=num_classes,
                        model_type='2D',
                        classification_mode=classification_mode,
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        n_trials=20,
                    )

                final_model_path = f"final_model_{classification_mode}_{feature_type}.h5"
                model.save(final_model_path)
                mlflow.log_artifact(final_model_path)
                models.append(model)

    processing_logger.info("All experiments completed successfully!")

if __name__ == "__main__":
    main()
