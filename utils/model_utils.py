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

from keras.models import Sequential
from keras.layers import (
    Conv1D, Conv2D, MaxPooling1D, MaxPooling2D,
    GlobalAveragePooling1D, GlobalAveragePooling2D,
    Dense, Dropout, BatchNormalization
)

 
# Initialize logger
model_logger = logging.getLogger("model_utils")


# ==========================
#  MODEL BUILDING UTILITIES
# ==========================

def build_cnn_model(input_shape, n_filters=32, dense_units=128, dropout_rate=0.3, num_classes=2, model_type='1D'):
    """
    Build and compile a CNN model.

    Args:
        input_shape: Shape of the input data.
        n_filters: Number of filters for the convolutional layers.
        dense_units: Number of units in the dense layer.
        dropout_rate: Dropout rate for regularization.
        num_classes: Number of output classes.
        model_type: '1D' for 1D CNN, '2D' for 2D CNN.

    Returns:
        Compiled CNN model.
    """
    model_logger.info(f"Building a {model_type} CNN model with input shape {input_shape}.")
    model = Sequential()

    if model_type == '1D':
        # 1D CNN layers
        model.add(Conv1D(n_filters, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(dropout_rate))

        model.add(Conv1D(n_filters * 2, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(dropout_rate))

        model.add(Conv1D(n_filters * 4, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(dropout_rate))

    elif model_type == '2D':
        # 2D CNN layers
        model.add(Conv2D(n_filters, (3, 3), activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        if input_shape[0] >= 2:
            model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(dropout_rate))

        model.add(Conv2D(n_filters * 2, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        if input_shape[0] >= 4:
            model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(dropout_rate))

        model.add(Conv2D(n_filters * 4, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(GlobalAveragePooling2D())
        model.add(Dropout(dropout_rate))

    else:
        raise ValueError("Invalid model_type. Must be '1D' or '2D'.")

    # Fully connected layers
    model.add(Dense(dense_units, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax'))

    # Compile the model
    loss = 'binary_crossentropy' if num_classes == 1 else 'categorical_crossentropy'
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    model_logger.info(f"{model_type} CNN model built and compiled successfully.")
    return model


# ===============================
#  HYPERPARAMETER OPTIMIZATION
# ===============================

def optimize_cnn_model(trial, input_shape, num_classes, X_train, y_train, X_val, y_val, model_type='1D'):
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
        model_type: Type of model ('1D' or '2D').

    Returns:
        Best validation accuracy.
    """
    n_filters = trial.suggest_int("n_filters", 16, 64, step=16)
    dense_units = trial.suggest_int("dense_units", 64, 256, step=64)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)

    model = build_cnn_model(input_shape, n_filters, dense_units, dropout_rate, num_classes, model_type=model_type)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=0)

    val_accuracy = max(history.history['val_accuracy'])
    return val_accuracy


def run_optuna_optimization(model_type, input_shape, num_classes, X_train, y_train, X_val, y_val, n_trials=20):
    """
    Run Optuna optimization for a given model type.

    Args:
        model_type: Type of model to optimize ('1D' or '2D').
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
        return optimize_cnn_model(trial, input_shape, num_classes, X_train, y_train, X_val, y_val, model_type)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    model_logger.info(f"Best trial for {model_type} CNN: {study.best_trial.params}")
    return study.best_trial.params


# ============================
#  DATASET PREPARATION UTILS
# ============================
from sklearn.model_selection import train_test_split
import numpy as np

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
    model_logger.info("Splitting dataset into training, validation, and test sets...")
    
    # Check for minimum class size
    class_counts = np.sum(y, axis=0) if len(y.shape) > 1 else np.bincount(y)
    if np.any(class_counts < 2):
        model_logger.warning("Some classes have fewer than 2 samples. Stratification will be disabled.")
        stratify_train = None
        stratify_test = None
    else:
        stratify_train = y
        stratify_test = y
    
    # Split training and test data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, stratify=stratify_train, random_state=random_state
    )
    
    # Split validation and test data
    class_counts_temp = np.sum(y_temp, axis=0) if len(y_temp.shape) > 1 else np.bincount(y_temp)
    if np.any(class_counts_temp < 2):
        model_logger.warning("Some classes in the temporary test set have fewer than 2 samples. Stratification will be disabled for the validation split.")
        stratify_temp = None
    else:
        stratify_temp = y_temp

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=validation_size, stratify=stratify_temp, random_state=random_state
    )
    
    model_logger.info("Dataset split completed.")
    return X_train, X_val, X_test, y_train, y_val, y_test
