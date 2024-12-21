import os
import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import logging
from utils.augmentation import add_noise, shift, stretch, pitch_shift  # Ensure augmentation functions are imported
from keras.utils import to_categorical, normalize


from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE



# Initialize logger
processing_logger = logging.getLogger("audio_processing")


def process_audio_file(soundDir, audio_files_path, df_filtered):
    """
    Process a single audio file: extract MFCC features and augment with noise, stretching, and shifting.

    Args:
        soundDir: Filename of the audio file.
        audio_files_path: Path to the directory containing audio files.
        df_filtered: Filtered DataFrame containing patient diagnosis and metadata.

    Returns:
        Tuple containing features (X_local) and labels (y_local).
    """
    X_local = []
    y_local = []
    features = 52  # Number of MFCC features

    try:
        # Extract patient ID and disease from filename and DataFrame
        patient_id = int(soundDir.split('_')[0])
        disease = df_filtered.loc[df_filtered['Patient number'] == patient_id, 'Diagnosis'].values[0]

        # Load audio file
        data_x, sampling_rate = librosa.load(os.path.join(audio_files_path, soundDir), sr=None)
        mfccs = np.mean(librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=features).T, axis=0)
        X_local.append(mfccs)
        y_local.append(disease)

        # Apply augmentations
        augmentations = [
            (add_noise, {"x": 0.001}),
            (shift, {"x": 1600}),
            (stretch, {"rate": 1.2}),
            (pitch_shift, {"rate": 3}),
        ]

        for func, kwargs in augmentations:
            augmented_data = func(data_x, **kwargs)
            mfccs_augmented = np.mean(librosa.feature.mfcc(y=augmented_data, sr=sampling_rate, n_mfcc=features).T, axis=0)
            X_local.append(mfccs_augmented)
            y_local.append(disease)

    except Exception as e:
        processing_logger.error(f"Error processing file {soundDir}: {e}")

    return X_local, y_local


def mfccs_feature_extraction(audio_files_path, df_filtered, n_jobs=-1):
    """
    Extract MFCC features from audio data and augment with noise, stretching, and shifting in parallel.

    Args:
        audio_files_path: Path to the directory containing audio files.
        df_filtered: Filtered DataFrame containing patient diagnosis and metadata.
        n_jobs: Number of parallel jobs (-1 to use all available cores).

    Returns:
        X_data: Array of features extracted from the audio files.
        y_data: Array of target labels.
    """
    processing_logger.info(f"Processing audio files in: {audio_files_path}")
    files = [file for file in os.listdir(audio_files_path) if file.endswith('.wav') and file[:3] not in ['103', '108', '115']]
    #files = files[:40]  # DEBUG limit, adjust as needed

    # Use Parallel and delayed to process files in parallel
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_audio_file)(file, audio_files_path, df_filtered) for file in tqdm(files, desc="Processing audio files")
    )

    # Flatten results
    X_, y_ = [], []
    for X_local, y_local in results:
        X_.extend(X_local)
        y_.extend(y_local)

    X_data = np.array(X_)
    y_data = np.array(y_)
    processing_logger.info("MFCC feature extraction and augmentation complete.")
    return X_data, y_data


def prepare_dataset_with_gru(df_filtered, audio_files_path):
    """
    Prepare the dataset using the GRU pipeline.

    Args:
        df_filtered: Filtered DataFrame containing patient diagnosis and metadata.
        audio_files_path: Path to the directory containing audio files.

    Returns:
        X: Feature data.
        y: Encoded labels.
        le: Label encoder.
    """
    processing_logger.info("Preparing dataset with GRU pipeline.")
    X, y = mfccs_feature_extraction(audio_files_path, df_filtered)
    le = LabelEncoder()
    y = to_categorical(le.fit_transform(np.array(y)))
    processing_logger.info("Dataset preparation with GRU pipeline complete.")
    return X, y, le


def process_audio_metadata(folder_path):
    """
    Extract audio metadata from filenames.

    Args:
        folder_path: Path to the folder containing metadata files.

    Returns:
        Metadata DataFrame.
    """
    processing_logger.info("Extracting audio metadata from filenames.")
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            parts = filename.split('_')
            try:
                data.append({
                    'Patient number': int(parts[0]),
                    'Recording index': parts[1],
                    'Chest location': parts[2],
                    'Acquisition mode': parts[3],
                    'Recording equipment': parts[4].split('.')[0],
                })
            except (IndexError, ValueError) as e:
                processing_logger.warning(f"Skipping file {filename}: {e}")
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
    Filter and sample the dataset.
    
    Args:
        df: Input DataFrame containing diagnosis data.
        mode: Specify 'binary' for Normal/Abnormal or 'multiclass' for original labels.
        
    Returns:
        Filtered and processed DataFrame.
    """
    processing_logger.info("Filtering and sampling the dataset.")
    if mode == 'binary':
        df['Diagnosis'] = df['Diagnosis'].apply(lambda x: 'Abnormal' if x != 'Healthy' else 'Normal')
    # Else, keep original multiclass labels
    df = df.sort_values('Patient number').reset_index(drop=True)
    processing_logger.info(f"Filtering and sampling complete with mode={mode}.")
    return df


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



def augment_data(X, y):
    """Apply data augmentation to increase dataset size."""
    processing_logger.info("Applying data augmentation.")
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X)
    processing_logger.info("Data augmentation setup complete.")
    return datagen

def preprocess_file(row, audio_files_path, mode):
    """Preprocess a single audio file."""
    file_path = os.path.join(audio_files_path, row['audio_file_name'])
    feature = preprocessing(file_path, mode)
    label = row['Diagnosis']
    return feature, label

def prepare_dataset_parallel(df, audio_files_path, mode):
    """Prepare the dataset by extracting features from audio files in parallel."""
    processing_logger.info(f"Preparing dataset using {mode} features in parallel.")
    results = Parallel(n_jobs=-1)(delayed(preprocess_file)(row, audio_files_path, mode) for _, row in tqdm(df.iterrows(), total=len(df)))

    X, y = zip(*results)
    X = np.array(X)
    X = np.expand_dims(X, axis=-1)  # Add channel dimension
    X = normalize(X, axis=1)
    le = LabelEncoder()
    y = to_categorical(le.fit_transform(np.array(y)))

    processing_logger.info(f"Dataset preparation using {mode} complete.")
    return X, y, le

def preprocessing(audio_file, mode):
    """Preprocess audio file by resampling, padding/truncating, and extracting features."""
    sr_new = 16000  # Resample audio to 16 kHz
    x, sr = librosa.load(audio_file, sr=sr_new)

    # Padding or truncating to 5 seconds (5 * sr_new samples)
    max_len = 5 * sr_new
    if x.shape[0] < max_len:
        x = np.pad(x, (0, max_len - x.shape[0]))
    else:
        x = x[:max_len]

    # Extract features
    if mode == 'mfcc':
        feature = librosa.feature.mfcc(y=x, sr=sr_new, n_mfcc=20)  # Ensure consistent shape
    elif mode == 'log_mel':
        feature = librosa.feature.melspectrogram(y=x, sr=sr_new, n_mels=20, fmax=8000)  # Match n_mels to 20
        feature = librosa.power_to_db(feature, ref=np.max)

    return feature

def prepare_dataset(df, audio_files_path, mode):
    """Prepare the dataset by extracting features from audio files."""
    processing_logger.info(f"Preparing dataset using {mode} features.")
    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        file_path = os.path.join(audio_files_path, row['audio_file_name'])
        feature = preprocessing(file_path, mode)
        X.append(feature)
        y.append(row['Diagnosis'])
        del feature  # Free memory after processing each file
        gc.collect()

    X = np.array(X)
    X = np.expand_dims(X, axis=-1)  # Add channel dimension
    X = normalize(X, axis=1)
    le = LabelEncoder()
    y = to_categorical(le.fit_transform(np.array(y)))
    processing_logger.info(f"Dataset preparation using {mode} complete.")
    return X, y, le