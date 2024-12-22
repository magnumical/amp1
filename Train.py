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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs from TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
data_logger = logging.getLogger("data_pipeline")
train_logger = logging.getLogger("train")
processing_logger = logging.getLogger("data_processing")
model_logger = logging.getLogger("model_training")

# Dataset and Paths
AUDIO_FILES_PATH = 'D://github//AmpleHealth//data//Respiratory_Sound_Database//audio_and_txt_files'
METADATA_PATH = 'D://github//AmpleHealth//data//Respiratory_Sound_Database//audio_and_txt_files'

def save_dataset(X, y, mode, output_dir="c"):
    """
    Save the processed X and y to .npy files.
    
    Args:
        X: Processed features.
        y: Processed labels.
        mode: Mode for which the dataset is processed.
        output_dir: Directory to save the .npy files.
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


def load_or_process_dataset(df_filtered, audio_files_path, mode, output_dir="processed_datasets"):

    #output_dir = os.path.abspath(output_dir)
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

        if mode == 'augmented':
            X, y, le = prepare_dataset_augmented(df_filtered, audio_files_path)
        else:
            X, y, le = prepare_dataset_parallel(df_filtered, audio_files_path, mode=mode)



        # Save the processed data and LabelEncoder
        np.save(X_path, X)
        np.save(y_path, y)
        processing_logger.info(f"Saved processed dataset and LabelEncoder for mode '{mode}' to {output_dir}")

    le = LabelEncoder()
    return X, y, le


def main():
    data_logger.info("Starting data pipeline.")

    # Step 1: Load and preprocess data
    data_logger.info("Loading and preprocessing data...")
    df = load_data()
    audio_metadata = process_audio_metadata(METADATA_PATH)
    df_all = merge_datasets(audio_metadata, df)

    # Define classification modes and feature types
    classification_modes = [ 'multi', 'binary']#
    feature_types = [ 'augmented','mfcc', 'log_mel'] #, 

    for classification_mode in classification_modes:
        # Preprocess dataset for binary or multi-class classification
        df_filtered = filter_and_sample_data(df_all, mode=classification_mode)

        for feature_type in feature_types:
            processing_logger.info(f"Preparing dataset for {classification_mode} classification with {feature_type} features.")
            
            # Load or process dataset
            X, y, le = load_or_process_dataset(df_filtered, AUDIO_FILES_PATH, feature_type, output_dir=f"processed_datasets/{classification_mode}")

            # Log input dimensions
            processing_logger.info(f"Input data dimensions for {feature_type}: {X.shape}")
            processing_logger.info(f"Output data dimensions for {feature_type}: {y.shape}")

            # Split dataset
            processing_logger.info("Splitting dataset...")
            X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

            # Check for class balance
            unique_classes, class_counts = np.unique(np.argmax(y_train, axis=1), return_counts=True)
            processing_logger.info(f"Class distribution before oversampling: {dict(zip(unique_classes, class_counts))}")

            try:
                X_train, y_train = oversample_data(X_train, y_train)
                unique_classes, class_counts = np.unique(np.argmax(y_train, axis=1), return_counts=True)
                processing_logger.info(f"Class distribution after oversampling: {dict(zip(unique_classes, class_counts))}")
            except ValueError as e:
                processing_logger.warning(f"SMOTE skipped: {e}")


            # Log dimensions after preprocessing
            processing_logger.info(f"Training data dimensions for {feature_type}: X_train={X_train.shape}, y_train={y_train.shape}")
            processing_logger.info(f"Validation data dimensions for {feature_type}: X_val={X_val.shape}, y_val={y_val.shape}")
            processing_logger.info(f"Test data dimensions for {feature_type}: X_test={X_test.shape}, y_test={y_test.shape}")

            # Train and optimize model
            model_logger.info(f"Running optimization for {feature_type} mode...")
            
            if feature_type == 'augmented':  # Train 1D CNN for GRU features
                X_train = np.expand_dims(X_train, axis=-1)
                X_val = np.expand_dims(X_val, axis=-1)
                X_test = np.expand_dims(X_test, axis=-1)

                model_logger.info(f"Updated 1D CNN Input dimensions: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")

                best_params = run_optuna_optimization(
                    model_type="1D",
                    input_shape=X_train.shape[1:],
                    num_classes=y_train.shape[1],
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    n_trials=20
                )
                best_model = build_cnn_model(
                    input_shape=X_train.shape[1:],
                    n_filters=best_params["n_filters"],
                    dense_units=best_params["dense_units"],
                    dropout_rate=best_params["dropout_rate"],
                    num_classes=y_train.shape[1],
                    model_type="1D"
                )
            else:  # Train 2D CNN for MFCC and Log-Mel
                best_params = run_optuna_optimization(
                    model_type="2D",
                    input_shape=X_train.shape[1:],
                    num_classes=y_train.shape[1],
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    n_trials=20
                )
                best_model = build_cnn_model(
                    input_shape=X_train.shape[1:],
                    n_filters=best_params["n_filters"],
                    dense_units=best_params["dense_units"],
                    dropout_rate=best_params["dropout_rate"],
                    num_classes=y_train.shape[1],
                    model_type="2D"
                )

            model_logger.info(f"Model input shape: {X_train.shape[1:]}")
            model_logger.info(f"Number of output classes: {y_train.shape[1]}")

            # Train and save the model
            best_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
            model_path = f"./best_model_{classification_mode}_{feature_type}.h5"
            best_model.save(model_path)
            mlflow.log_artifact(model_path)

            # Evaluate model
            y_pred = best_model.predict(X_test)
            log_metrics(y_test, y_pred, f"{classification_mode}_{feature_type}")

    data_logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()

