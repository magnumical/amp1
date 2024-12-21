import numpy as np
import logging
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, GRU, Input, add
from keras.optimizers import Adamax
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import optuna

from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from imblearn.over_sampling import SMOTE



from tensorflow.keras.models import Sequential, Model, load_model

from tensorflow.keras.layers import Conv1D, Conv2D, SeparableConv1D, MaxPooling1D, MaxPooling2D
from tensorflow.keras.layers import Input, add, Flatten, Dense, BatchNormalization, Dropout, LSTM, GRU
from tensorflow.keras.layers import GlobalMaxPooling1D, GlobalMaxPooling2D, Activation, LeakyReLU, ReLU

from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# Initialize logger
model_logger = logging.getLogger("model_utils")


# ==========================
#  MODEL BUILDING UTILITIES
# ==========================

def build_cnn_model(input_shape, n_filters=32, dense_units=128, dropout_rate=0.3, num_classes=2):
    """
    Build and compile a CNN model.

    Args:
        input_shape: Shape of the input data.
        n_filters: Number of filters for the convolutional layers.
        dense_units: Number of units in the dense layer.
        dropout_rate: Dropout rate for regularization.
        num_classes: Number of output classes.

    Returns:
        Compiled CNN model.
    """
    model_logger.info("Building CNN model.")
    model = Sequential([
        Conv2D(n_filters, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),

        Conv2D(n_filters * 2, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),

        GlobalAveragePooling2D(),
        Dense(dense_units, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model_logger.info("CNN model built and compiled successfully.")
    return model


def build_gru_model(input_shape, num_units=128, dropout_rate=0.3, num_classes=2):
    model_logger.info("Building GRU model.")
    Input_Sample = Input(shape=input_shape)

    # GRU architecture
    model_gru = GRU(num_units, return_sequences=True, activation='tanh')(Input_Sample)
    model_gru = GRU(num_units, return_sequences=False, activation='tanh')(model_gru)
    model_gru = Dropout(dropout_rate)(model_gru)

    # Fully connected layers
    dense_layer = Dense(num_units, activation='relu')(model_gru)
    dense_layer = Dropout(dropout_rate)(dense_layer)
    output_layer = Dense(num_classes, activation='softmax')(dense_layer)

    model = Model(inputs=Input_Sample, outputs=output_layer)
    model.compile(optimizer=Adamax(), loss='categorical_crossentropy', metrics=['accuracy'])
    model_logger.info("GRU model built and compiled successfully.")
    return model



# ===============================
#  HYPERPARAMETER OPTIMIZATION
# ===============================

def optimize_cnn_model(trial, input_shape, num_classes, X_train, y_train, X_val, y_val):
    """
    Optimize CNN model using Optuna.

    Args:
        trial: Optuna trial object.
        input_shape: Shape of the input data.
        num_classes: Number of output classes.
        X_train: Training data.
        y_train: Training labels.
        X_val: Validation data.
        y_val: Validation labels.

    Returns:
        Best validation accuracy.
    """
    n_filters = trial.suggest_int("n_filters", 16, 64, step=16)
    dense_units = trial.suggest_int("dense_units", 64, 256, step=64)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)

    model = build_cnn_model(input_shape, n_filters, dense_units, dropout_rate, num_classes)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=0)

    val_accuracy = max(history.history['val_accuracy'])
    return val_accuracy


def optimize_gru_model(trial, input_shape, num_classes, X_train, y_train, X_val, y_val):
    """
    Optimize GRU model using Optuna.

    Args:
        trial: Optuna trial object.
        input_shape: Shape of the input data.
        num_classes: Number of output classes.
        X_train: Training data.
        y_train: Training labels.
        X_val: Validation data.
        y_val: Validation labels.

    Returns:
        Best validation accuracy.
    """
    num_units = trial.suggest_int("num_units", 64, 256, step=32)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)

    model = build_gru_model(input_shape, num_units, dropout_rate, num_classes)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=0)

    val_accuracy = max(history.history['val_accuracy'])
    return val_accuracy


def run_optuna_optimization(model_type, input_shape, num_classes, X_train, y_train, X_val, y_val, n_trials=20):
    """
    Run Optuna optimization for a given model type.

    Args:
        model_type: Type of model to optimize ('cnn' or 'gru').
        input_shape: Shape of the input data.
        num_classes: Number of output classes.
        X_train: Training data.
        y_train: Training labels.
        X_val: Validation data.
        y_val: Validation labels.
        n_trials: Number of trials for Optuna optimization.

    Returns:
        Best hyperparameters.
    """
    def objective(trial):
        if model_type == "cnn":
            return optimize_cnn_model(trial, input_shape, num_classes, X_train, y_train, X_val, y_val)
        elif model_type == "gru":
            return optimize_gru_model(trial, input_shape, num_classes, X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    model_logger.info(f"Best trial for {model_type}: {study.best_trial.params}")
    return study.best_trial.params


# ============================
#  DATASET PREPARATION UTILS
# ============================

def split_dataset(X, y, test_size=0.3, validation_size=0.5, random_state=42):
    """
    Split dataset into training, validation, and test sets.

    Args:
        X: Feature data.
        y: Labels.
        test_size: Proportion of the data to reserve for testing.
        validation_size: Proportion of the test set to reserve for validation.
        random_state: Random seed.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=validation_size, stratify=y_temp, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test
