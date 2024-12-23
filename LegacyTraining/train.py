import os
import logging
import gc
from joblib import Parallel, delayed
import joblib
import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
import librosa
import librosa.display
import optuna
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.utils import to_categorical, normalize
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.layers import Conv1D, MaxPooling1D,GlobalAveragePooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from imblearn.over_sampling import SMOTE
from scipy.signal import butter, sosfilt
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
data_logger = logging.getLogger("data_loading")
processing_logger = logging.getLogger("data_processing")
model_logger = logging.getLogger("model_training")


def load_data(diagnosis_path='/kaggle/input/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv',
              demographic_path='/kaggle/input/respiratory-sound-database/demographic_info.txt'):
    """Load patient diagnosis and demographic data."""
    data_logger.info("Loading patient diagnosis and demographic data.")
    
    # Load diagnosis data
    diagnosis_df = pd.read_csv(diagnosis_path, 
                               names=['Patient number', 'Diagnosis'])

    # Load demographic data
    patient_df = pd.read_csv(demographic_path, 
                             names=['Patient number', 'Age', 'Sex', 'Adult BMI (kg/m2)', 'Child Weight (kg)', 'Child Height (cm)'],
                             delimiter=' ')

    data_logger.info("Data successfully loaded.")
    
    # Merge and return
    return pd.merge(left=patient_df, right=diagnosis_df, how='left')


def process_audio_metadata(folder_path):
    """Extract audio metadata from filenames."""
    processing_logger.info("Extracting audio metadata from filenames.")
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            parts = filename.split('_')
            data.append({
                'Patient number': int(parts[0]),
                'Recording index': parts[1],
                'Chest location': parts[2],
                'Acquisition mode': parts[3],
                'Recording equipment': parts[4].split('.')[0]
            })
    processing_logger.info("Audio metadata extraction complete.")
    return pd.DataFrame(data)


def merge_datasets(df1, df2):
    """Merge metadata and diagnosis data."""
    processing_logger.info("Merging metadata and diagnosis data.")
    merged_df = pd.merge(left=df1, right=df2, how='left').sort_values('Patient number').reset_index(drop=True)
    merged_df['audio_file_name'] = merged_df.apply(lambda row: f"{row['Patient number']}_{row['Recording index']}_{row['Chest location']}_{row['Acquisition mode']}_{row['Recording equipment']}.wav", axis=1)
    processing_logger.info("Merging complete.")
    return merged_df



def filter_and_sample_data(df, mode='binary'):
    """
    Filter and sample the dataset for binary or multi-class classification.

    Returns filtered and processed DataFrame.
    """
    processing_logger.info(f"Filtering and sampling the dataset for {mode} classification.")
    
    if mode == 'binary':
        # Binary classification: Normal vs. Abnormal
        df['Diagnosis'] = df['Diagnosis'].apply(lambda x: 'Normal' if x == 'Healthy' else 'Abnormal')
    elif mode == 'multi':
        # Multi-class classification: Group classes
        # I grouped disease based on their similarities
        processing_logger.info("Grouping classes for multi-class classification.")
        df['Diagnosis'] = df['Diagnosis'].replace({
            'Healthy': 'Normal',
            'COPD': 'Chronic Respiratory Diseases',
            'Asthma': 'Chronic Respiratory Diseases',
            'URTI': 'Respiratory Infections',
            'Bronchiolitis': 'Respiratory Infections',
            'LRTI': 'Respiratory Infections',
            'Pneumonia': 'Respiratory Infections',
            'Bronchiectasis': 'Respiratory Infections'
        })

    # Filter out rare classes with fewer than 5 samples
    class_counts = df['Diagnosis'].value_counts()
    valid_classes = class_counts[class_counts >= 5].index
    df = df[df['Diagnosis'].isin(valid_classes)].reset_index(drop=True)

    processing_logger.info(f"Filtered classes: {df['Diagnosis'].unique()}")
    processing_logger.info(f"Filtering and sampling complete with mode={mode}.")
    return df


def prepare_dataset_augmented(df_filtered, audio_files_path, classification_mode):
    """Prepare the dataset for augmented features. it will be 1D array"""
    processing_logger.info("Preparing dataset with AUGMENTED pipeline.")
    
    # Extract features and labels
    X, y = mfccs_feature_extraction(audio_files_path, df_filtered)
    
    # Apply label encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(np.array(y))  # Encode labels to integers

    if classification_mode == "binary":
        # Use single column with 0 and 1 for binary classification
        processing_logger.info("Binary classification mode: Using single column labels (0/1).")
        y_processed = y_encoded  # No one-hot encoding
    else:
        # One-hot encode labels for multi-class classification
        processing_logger.info("Multi-class classification mode: Applying one-hot encoding.")
        y_processed = to_categorical(y_encoded)

        # Log the mapping of one-hot encoding to class labels
        print("One-hot encoding mapping:")
        for idx, label in enumerate(le.classes_):
            print(f"{idx} -> {label}")
    
    processing_logger.info("Dataset preparation with augmented pipeline complete.")
    return X, y_processed, le


def mfccs_feature_extraction(audio_files_path, df_filtered, n_jobs=-1):
    """
    Make the process of MFCC feature extraction faster by running jobs in-parallel
    
    Returns array of features extracted from the audio files and Array of target labels.
    """
    processing_logger.info(f"Processing audio files in: {audio_files_path}")
    files = [file for file in os.listdir(audio_files_path) if file.endswith('.wav') and file[:3] not in ['103', '108', '115']]
   
    #files = files[:30] ## DEBUG

    # Use Parallel and delayed to process files in parallel
    results = Parallel(n_jobs=n_jobs, backend="loky")(delayed(process_audio_file)(file, audio_files_path, df_filtered) for file in tqdm(files, desc="Processing audio files"))

    # Flatten results
    X_ = []
    y_ = []
    for X_local, y_local in results:
        X_.extend(X_local)
        y_.extend(y_local)

    X_data = np.array(X_)
    y_data = np.array(y_)
    processing_logger.info("MFCC feature extraction and augmentation complete.")
    return X_data, y_data


def process_audio_file(soundDir, audio_files_path, df_filtered):
    """
    Process a single audio file: extract MFCC features and augment with noise, stretching, and shifting.
    
    """
    X_local = []
    y_local = []
    features = 52

    # Extract patient ID and disease from filename and DataFrame
    patient_id = int(soundDir.split('_')[0])
    disease = df_filtered.loc[df_filtered['Patient number'] == patient_id, 'Diagnosis'].values[0]

    # Load audio file
    data_x, sampling_rate = librosa.load(os.path.join(audio_files_path, soundDir), sr=None)
    data_x = preprocess_audio(data_x, sampling_rate)  # Apply filtering

    
    mfccs = np.mean(librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=features).T, axis=0)
    X_local.append(mfccs)
    y_local.append(disease)

    # Data augmentation
    for augmentation in [add_noise, shift, stretch, pitch_shift]:
        if augmentation == add_noise:
            augmented_data = augmentation(data_x, 0.001)
        elif augmentation == shift:
            augmented_data = augmentation(data_x, 1600)
        elif augmentation == stretch:
            augmented_data = augmentation(data_x, 1.2)
        elif augmentation == pitch_shift:
            augmented_data = augmentation(data_x, sampling_rate, 3)

        mfccs_augmented = np.mean(librosa.feature.mfcc(y=augmented_data, sr=sampling_rate, n_mfcc=features).T, axis=0)
        X_local.append(mfccs_augmented)
        y_local.append(disease)

    return X_local, y_local


def add_noise(data,x):
    noise = np.random.randn(len(data))
    data_noise = data + x * noise
    return data_noise

def shift(data, x):
    return np.roll(data, int(x))

def stretch(data, rate):
    return librosa.effects.time_stretch(data, rate=rate)

def pitch_shift (data , sr, rate):
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=rate)




def prepare_dataset_parallel(df, audio_files_path, mode, classification_mode):
    """Prepare the dataset by extracting features from audio files in parallel."""
    processing_logger.info(f"Preparing dataset using {mode} features in parallel.")
    results = Parallel(n_jobs=-1)(delayed(preprocess_file)(row, audio_files_path, mode) for _, row in tqdm(df.iterrows(), total=len(df)))

    X, y = zip(*results)
    X = np.array(X)
    X = np.expand_dims(X, axis=-1)  # Add channel dimension
    X = normalize(X, axis=1)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(np.array(y))  # Encode labels

    if classification_mode == "binary":
        # Use single column with 0 and 1 for binary classification
        processing_logger.info("Binary classification mode: Using single column labels (0/1).")
        y = y_encoded  # No one-hot encoding
    else:
        # One-hot encode labels for multi-class classification
        processing_logger.info("Multi-class classification mode: Applying one-hot encoding.")
        y = to_categorical(y_encoded)

    processing_logger.info(f"Dataset preparation using {mode} complete.")
    return X, y, le

def preprocess_file(row, audio_files_path, mode):
    """Preprocess a single audio file."""
    file_path = os.path.join(audio_files_path, row['audio_file_name'])
    feature = preprocessing(file_path, mode)
    label = row['Diagnosis']
    return feature, label
    
def preprocessing(audio_file, mode):
    """Preprocess audio file by resampling, padding/truncating, and extracting features."""
    sr_new = 16000  # Resample audio to 16 kHz
    x, sr = librosa.load(audio_file, sr=sr_new)
    x = preprocess_audio(x, sr)
    # Padding or truncating to 5 seconds (5 * sr_new samples)
    max_len = 5 * sr_new
    if x.shape[0] < max_len:
        x = np.pad(x, (0, max_len - x.shape[0]))
    else:
        x = x[:max_len]

    # Extract features
    # I understand the common choice for n_mfcc is 13, but here i assumed we need to capture more informationm, therefore I choose 20.
    if mode == 'mfcc':
        feature = librosa.feature.mfcc(y=x, sr=sr_new, n_mfcc=20)  # Ensure consistent shape
    elif mode == 'log_mel':
        feature = librosa.feature.melspectrogram(y=x, sr=sr_new, n_mels=20, fmax=8000)  # Match n_mels to 20
        feature = librosa.power_to_db(feature, ref=np.max)

    return feature

def oversample_data(X, y):
    """Apply SMOTE to balance classes."""
    processing_logger.info("Applying SMOTE to balance classes.")
    
    # Save the original shape of features
    original_shape = X.shape[1:]  
    
    # Flatten for SMOTE processing
    X = X.reshape((X.shape[0], -1))
    
    # Convert one-hot encoded labels to integers
    y = np.argmax(y, axis=1)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Reshape back to the original dimensions
    X_resampled = X_resampled.reshape((-1, *original_shape))
    
    # Convert labels back to one-hot encoding
    y_resampled = to_categorical(y_resampled)
    
    processing_logger.info("SMOTE oversampling complete.")
    return X_resampled, y_resampled



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


def log_metrics(y_true, y_pred, mode):
    """Log evaluation metrics."""
    precision = classification_report(y_true, y_pred, output_dict=True)['weighted avg']['precision']
    recall = classification_report(y_true, y_pred, output_dict=True)['weighted avg']['recall']
    f1_score = classification_report(y_true, y_pred, output_dict=True)['weighted avg']['f1-score']

    mlflow.log_metric(f"{mode}_precision", precision)
    mlflow.log_metric(f"{mode}_recall", recall)
    mlflow.log_metric(f"{mode}_f1_score", f1_score)



def track_experiment_with_mlflow_and_optuna(mode, num_classes, model_type='1D', classification_mode='binary'):
    """
    Optimize hyperparameters using Optuna and track experiments with MLflow.

    mode: Feature extraction mode (e.g., 'augmented', 'mfcc', 'log_mel').
    num_classes: Number of classes for classification.
    model_type: Type of model ('1D' for Conv1D, '2D' for Conv2D).
    classification_mode: 'binary' for binary classification, 'multi' for multi-class classification.
    """
    def objective(trial):
        with mlflow.start_run(nested=True):  # Start a new MLflow run for each trial
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
                monitor='val_loss',  # Monitor validation loss
                patience=5,          # Stop training after 5 epochs with no improvement
                restore_best_weights=True
            )

            # Train the model
            history = model.fit(
                X_train, y_train, 
                validation_data=(X_val, y_val), 
                epochs=50,  # Allow a larger max epoch since EarlyStopping will handle early termination
                batch_size=32, 
                callbacks=[early_stopping], 
                verbose=0
            )

            # Log hyperparameters and metrics to MLflow
            mlflow.log_params({
                'n_filters': n_filters,
                'dense_units': dense_units,
                'dropout_rate': dropout_rate,
                'learning_rate': learning_rate,
                'model_type': model_type,
                'classification_mode': classification_mode
            })
            mlflow.log_metric("best_val_accuracy", max(history.history['val_accuracy']))

            # Save training and validation loss curves
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
    study.optimize(objective, n_trials=20)

    # Retrieve best trial and log results
    best_trial = study.best_trial
    model_logger.info(f"Best Trial for {mode} ({model_type}): {best_trial.params}")

    # Build the best model (already compiled in build_model)
    best_model = build_model(
        input_shape=X_train.shape[1:], 
        n_filters=best_trial.params['n_filters'], 
        dense_units=best_trial.params['dense_units'], 
        dropout_rate=best_trial.params['dropout_rate'], 
        num_classes=num_classes,
        model_type=model_type,
        classification_mode=classification_mode
    )

    # Train the best model with EarlyStopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    best_model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=50, batch_size=32, 
        callbacks=[early_stopping], 
        verbose=1
    )

    # Save the best model
    best_model_path = f"best_model_{mode}_{model_type}.h5"
    best_model.save(best_model_path)
    mlflow.log_artifact(best_model_path)
    model_logger.info(f"Best model for {mode} ({model_type}) saved successfully.")

    return best_model

def log_class_distribution(y, message):
    """Log the class distribution."""
    if y.ndim == 1:  # Binary classification (1D array of 0s and 1s)
        unique, counts = np.unique(y, return_counts=True)
    else:  # Multi-class classification (2D one-hot encoded array)
        unique, counts = np.unique(np.argmax(y, axis=1), return_counts=True)

    class_distribution = dict(zip(unique, counts))
    processing_logger.info(f"{message} Class Distribution: {class_distribution}")


def preprocess_audio(audio, sr):
    """
    Apply a bandpass filter to audio data.
    
    """
    # Define cutoff frequencies
    low_cutoff = 50  # 50 Hz
    high_cutoff = min(5000, sr / 2 - 1)  # Ensure it is below Nyquist frequency

    if low_cutoff >= high_cutoff:
        raise ValueError(
            f"Invalid filter range: low_cutoff={low_cutoff}, high_cutoff={high_cutoff} for sampling rate {sr}"
        )

    # Design a bandpass filter
    sos = butter(N=10, Wn=[low_cutoff, high_cutoff], btype='band', fs=sr, output='sos')

    # Apply the filter
    filtered_audio = sosfilt(sos, audio)
    return filtered_audio


def generate_random_audio_data(samples=20000, feature_dim=20):
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
        classification_mode='binary'
    )

    print("[DEBUG] Training the model...")
    # Train the model with a single epoch for testing
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=32)

    print("[DEBUG] Evaluating the model...")
    results = model.evaluate(X_test, y_test)
    print(f"[DEBUG] Test evaluation results: {results}")


def main():
    # how to run:
    #  python legacy/test.py --metadata_path data/Respiratory_Sound_Database/audio_and_txt_files --audio_files_path data/Respiratory_Sound_Database/audio_and_txt_files --demographic_path data/demographic_info.txt --diagnosis_path data/Respiratory_Sound_Database/patient_diagnosis.csv --classification_modes binary --feature_types mfcc 

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run the respiratory sound analysis pipeline.")
    parser.add_argument("--metadata_path", type=str, default="/kaggle/input/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files", help="Path to the metadata directory.")
    parser.add_argument("--audio_files_path", type=str, default="/kaggle/input/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files", help="Path to the directory containing audio files.")
    parser.add_argument("--demographic_path", type=str, default="/kaggle/input/respiratory-sound-database/demographic_info.txt", help="Path to the demographic info file.")
    parser.add_argument("--diagnosis_path", type=str, default="/kaggle/input/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv", help="Path to the patient diagnosis CSV file.")
    parser.add_argument("--tracking_uri", type=str, default="./mlruns", help="MLflow tracking URI.")
    parser.add_argument("--classification_modes", type=str, nargs='+', default=['multi', 'binary'], help="Classification modes to run (default: all modes). Options: 'binary', 'multi'.")
    parser.add_argument("--feature_types", type=str, nargs='+', default=['mfcc', 'log_mel', 'augmented'], help="Feature types to use (default: all types). Options: 'mfcc', 'log_mel', 'augmented'.")
    parser.add_argument("--debug", action='store_true', help="Run in debug mode with random test data.")
    args = parser.parse_args()

    if args.debug:
        test_model()
        return
    # Assign arguments to variables
    metadata_path = args.metadata_path
    audio_files_path = args.audio_files_path
    demographic_path = args.demographic_path
    diagnosis_path = args.diagnosis_path


    # Set MLflow tracking URI
    mlflow.set_tracking_uri(args.tracking_uri)

    metadata_path = args.metadata_path
    audio_files_path = args.audio_files_path

    data_logger.info("Starting data pipeline.")
    df = load_data(demographic_path=demographic_path, diagnosis_path=diagnosis_path)
    audio_metadata = process_audio_metadata(audio_files_path)
    df_all = merge_datasets(audio_metadata, df)

    # Use user-specified or default classification modes and feature types
    classification_modes = args.classification_modes
    feature_types = args.feature_types
    models = []

    for classification_mode in classification_modes:
        # Preprocess dataset for binary or multi-class classification
        df_filtered = filter_and_sample_data(df_all, mode=classification_mode)
        processing_logger.info(f"Dataset shape for {classification_mode} mode: {df_filtered.shape}")

        for feature_type in feature_types:
            processing_logger.info(f"Running experiment for {classification_mode} classification with {feature_type} features.")
            global X_train, X_val, X_test, y_train, y_val, y_test

            # Prepare the dataset
            if feature_type == 'augmented':
                X, y, le = prepare_dataset_augmented(
                    df_filtered, 
                    audio_files_path,
                    classification_mode=classification_mode
                )
            else:
                X, y, le = prepare_dataset_parallel(
                    df_filtered, 
                    audio_files_path,
                    mode=feature_type,
                    classification_mode=classification_mode
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
            if classification_mode == "binary":
                num_classes = 1  # Single output for binary classification
            else:
                num_classes = y_train.shape[1]  # Number of classes for multi-class

            # Train and save model
            with mlflow.start_run(run_name=f"Experiment_{classification_mode}_{feature_type}", nested=True):
                if feature_type == 'augmented':
                    # Expand dimensions for 1D CNN input
                    X_train = np.expand_dims(X_train, axis=-1)
                    X_val = np.expand_dims(X_val, axis=-1)
                    X_test = np.expand_dims(X_test, axis=-1)

                    # Optimize and train 1D CNN
                    model = track_experiment_with_mlflow_and_optuna(
                        mode=feature_type,
                        num_classes=num_classes,
                        model_type='1D',  # Specify 1D CNN for GRU features
                        classification_mode=classification_mode
                    )
                else:
                    # Optimize and train CNN models for MFCC and MEL
                    model = track_experiment_with_mlflow_and_optuna(
                        mode=feature_type,
                        num_classes=num_classes,
                        model_type='2D',  # Specify 2D CNN for MFCC and Log-Mel
                        classification_mode=classification_mode
                    )

                # Save final model
                final_model_path = f"final_model_{classification_mode}_{feature_type}.h5"
                model.save(final_model_path)
                mlflow.log_artifact(final_model_path)
                models.append(model)

    processing_logger.info("All experiments completed successfully!")


if __name__ == "__main__":
    main()
