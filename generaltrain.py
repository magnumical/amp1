import logging
import numpy as np
from utils.data_loader import load_data, process_audio_metadata
from utils.audioprocessing import *
from utils.model_utils import * 

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

def main():
    data_logger.info("Starting data pipeline.")

    # Step 1: Load and preprocess data
    data_logger.info("Loading and preprocessing data...")
    df = load_data()
    audio_metadata = process_audio_metadata(METADATA_PATH)
    df_all = merge_datasets(audio_metadata, df)
    df_filtered = filter_and_sample_data(df_all)

    # Modes to process
    modes = ['gru', 'mfcc', 'log_mel']

    for mode in modes:
        processing_logger.info(f"Preparing dataset for mode: {mode}")

        if mode == 'gru':
            X, y, le = prepare_dataset_with_gru(df_filtered, AUDIO_FILES_PATH)
        else:
            X, y, le = prepare_dataset_parallel(df_filtered, AUDIO_FILES_PATH, mode=mode)

        # Split dataset
        processing_logger.info("Splitting dataset...")
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

        # Check for class balance
        unique_classes, class_counts = np.unique(np.argmax(y_train, axis=1), return_counts=True)
        processing_logger.info(f"Classes in y_train: {unique_classes}, Counts: {class_counts}")
        if len(unique_classes) <= 1:
            raise ValueError(f"Insufficient class diversity in y_train for {mode} mode.")

        # Oversample for GRU models
        if mode == 'gru':
            try:
                X_train, y_train = oversample_data(X_train, y_train)
            except ValueError as e:
                processing_logger.warning(f"SMOTE skipped: {e}")

        # Optimize and train model
        model_logger.info(f"Running optimization for {mode} mode...")
        if mode == 'gru':
            X_train = np.expand_dims(X_train, axis=1)
            X_val = np.expand_dims(X_val, axis=1)
            X_test = np.expand_dims(X_test, axis=1)

            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: optimize_gru_model(
                trial, X_train.shape[1:], y_train.shape[1], X_train, y_train, X_val, y_val
            ), n_trials=20)

            best_params = study.best_params
            model_logger.info(f"Best GRU Hyperparameters: {best_params}")

            best_model = build_gru_model(
                input_shape=X_train.shape[1:],
                num_units=best_params["num_units"],
                dropout_rate=best_params["dropout_rate"],
                num_classes=y_train.shape[1]
            )
        else:
            best_params = run_optuna_optimization(
                model_type="cnn", 
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
                num_classes=y_train.shape[1]
            )

        # Train and save the model
        best_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
        model_path = f"./best_model_{mode}.h5"
        best_model.save(model_path)
        mlflow.log_artifact(model_path)

        # Evaluate model
        y_pred = best_model.predict(X_test)
        log_metrics(y_test, y_pred, mode)

    data_logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
