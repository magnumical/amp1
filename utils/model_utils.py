import numpy as np
import logging
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, GRU, Input, add
from keras.optimizers import Adamax
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import optuna
import mlflow
import mlflow.keras
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt



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
from sklearn.model_selection import train_test_split
import numpy as np

 
# Initialize logger
model_logger = logging.getLogger("model_utils")


# ==========================
#  MODEL BUILDING UTILITIES
# ==========================

def build_model(input_shape, n_filters, dense_units, dropout_rate, num_classes, model_type='1D', classification_mode='binary'):
    """
    Build and compile a CNN model for 1D or 2D data.

    Returns CNN model.
    """
    print(f"Building the updated {model_type} CNN model with {classification_mode} classification.")
    model = Sequential()

    # Add convolutional layers based on the model type
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

    # Add fully connected layers
    model.add(Dense(dense_units, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Add output layer dynamically based on classification mode
    if classification_mode == 'binary':
        # Binary classification: Single unit with sigmoid activation
        model.add(Dense(1, activation='sigmoid'))
        loss_function = 'binary_crossentropy'
    else:
        # Multi-class classification: num_classes units with softmax activation
        model.add(Dense(num_classes, activation='softmax'))
        loss_function = 'categorical_crossentropy'

    # Compile the model
    model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
    print(f"{model_type} CNN model built and compiled successfully for {classification_mode} classification.")
    return model




def track_experiment_with_mlflow_and_optuna(
    mode,
    num_classes,
    model_type,
    classification_mode,
    X_train,
    y_train,
    X_val,
    y_val,
    n_trials=20,
):
    """
    Optimize hyperparameters using Optuna and track experiments with MLflow.

    Parameters:
    - mode: Feature extraction mode (e.g., 'augmented', 'mfcc', 'log_mel').
    - num_classes: Number of classes for classification.
    - model_type: Type of model ('1D' for Conv1D, '2D' for Conv2D).
    - classification_mode: 'binary' for binary classification, 'multi' for multi-class classification.
    - X_train, y_train: Training data and labels.
    - X_val, y_val: Validation data and labels.
    - n_trials: Number of Optuna trials.
    """
    def objective(trial):
        with mlflow.start_run(nested=True):
            # Hyperparameters to tune
            n_filters = trial.suggest_categorical('n_filters', [16, 32, 64])
            dense_units = trial.suggest_int('dense_units', 64, 256, step=32)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1)
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)

            # Build and compile the model
            model = build_model(
                input_shape=X_train.shape[1:], 
                n_filters=n_filters, 
                dense_units=dense_units, 
                dropout_rate=dropout_rate,
                num_classes=num_classes,
                model_type=model_type,
                classification_mode=classification_mode
            )

            # Define EarlyStopping callback
            early_stopping = EarlyStopping(
                monitor='val_loss', 
                patience=5,  
                restore_best_weights=True
            )

            # Train the model
            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0,
            )

            # Log hyperparameters and metrics to MLflow
            mlflow.log_params({
                'n_filters': n_filters,
                'dense_units': dense_units,
                'dropout_rate': dropout_rate,
                'learning_rate': learning_rate,
                'model_type': model_type,
                'classification_mode': classification_mode,
            })
            mlflow.log_metric("best_val_accuracy", max(history.history['val_accuracy']))

            # Save loss curves
            plt.figure()
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.legend()
            plt.title("Training and Validation Loss")
            loss_curve_path = f"loss_curve_{trial.number}_{model_type}.png"
            plt.savefig(loss_curve_path)
            mlflow.log_artifact(loss_curve_path)

            return max(history.history['val_accuracy'])

    # Start Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    # Retrieve the best trial and log results
    best_trial = study.best_trial
    model_logger.info(f"Best Trial for {mode} ({model_type}): {best_trial.params}")

    # Build and return the best model
    best_model = build_model(
        input_shape=X_train.shape[1:], 
        n_filters=best_trial.params['n_filters'], 
        dense_units=best_trial.params['dense_units'], 
        dropout_rate=best_trial.params['dropout_rate'], 
        num_classes=num_classes,
        model_type=model_type,
        classification_mode=classification_mode
    )

    # Train the best model
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
    )
    best_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1,
    )

    # Save the best model
    best_model_path = f"best_model_{mode}_{model_type}.h5"
    best_model.save(best_model_path)
    mlflow.log_artifact(best_model_path)
    model_logger.info(f"Best model for {mode} ({model_type}) saved successfully.")

    return best_model
