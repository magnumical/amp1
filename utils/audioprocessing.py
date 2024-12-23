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
from keras.utils import normalize
from scipy.signal import butter, sosfilt


from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE



# Initialize logger
processing_logger = logging.getLogger("audio_processing")

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

    