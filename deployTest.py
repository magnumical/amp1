import os
import logging
import numpy as np
import librosa
from sklearn.preprocessing import normalize
from tensorflow.keras.models import load_model
from scipy.signal import butter, sosfilt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("audio_classifier_test")

# Paths and Constants
MODEL_PATH = "./models"
FILE_PATH = "101_1b1_Al_sc_Meditron.wav"
MODELS = {
    "binary": {
        "augmented": "final_model_binary_augmented.h5",
        "log_mel": "final_model_binary_log_mel.h5",
        "mfcc": "final_model_binary_mfcc.h5",
    },
    "multi": {
        "augmented": "final_model_multi_augmented.h5",
        "log_mel": "final_model_multi_log_mel.h5",
        "mfcc": "final_model_multi_mfcc.h5",
    }
}
CLASS_NAMES = {
    "binary": ["Abnormal", "Normal"],
    "multi": ["Chronic Respiratory Diseases", "Normal", "Respiratory Infections"]
}


# Augmentation Functions
def add_noise(data, noise_factor=0.001):
    noise = np.random.randn(len(data))
    return data + noise_factor * noise

def shift(data, shift_factor=1600):
    return np.roll(data, shift_factor)

def stretch(data, rate=1.2):
    return librosa.effects.time_stretch(data, rate=rate)

def pitch_shift(data, sr, n_steps=3):
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=n_steps)



def filtering(audio, sr):
    """
    Apply a bandpass filter to audio data.
    
    Args:
        audio: The input audio signal.
        sr: The sampling rate of the audio.
        
    Returns:
        Filtered audio signal.
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


def preprocess_audio(audio_file, mode="augmented", input_shape=None):
    """
    Preprocess an audio file for classification by resampling, padding/truncating,
    and extracting features (e.g., MFCC, Log-Mel spectrogram, or Augmented features).

    Args:
        audio_file: Path to the audio file.
        mode: Feature extraction mode ('mfcc', 'log_mel', or 'augmented').
        input_shape: Expected input shape of the model for feature alignment.

    Returns:
        Extracted features as per the mode.
    """
    try:
        sr_new = 16000  # Resample audio to 16 kHz
        x, sr = librosa.load(audio_file, sr=sr_new)
        x = filtering(x, sr)
        logger.info(f"Loaded audio file '{audio_file}' with shape {x.shape} and sampling rate {sr}.")

        max_len = 5 * sr_new
        if x.shape[0] < max_len:
            x = np.pad(x, (0, max_len - x.shape[0]))
            logger.info(f"Audio padded to {max_len} samples.")
        else:
            x = x[:max_len]
            logger.info(f"Audio truncated to {max_len} samples.")

        # Handle each mode separately
        if mode == 'mfcc':
            feature = librosa.feature.mfcc(y=x, sr=sr_new, n_mfcc=20)  # Extract MFCC
            feature = normalize(feature, axis=1)

        elif mode == 'log_mel':
            mel_spec = librosa.feature.melspectrogram(y=x, sr=sr_new, n_mels=20, fmax=8000)
            feature = librosa.power_to_db(mel_spec, ref=np.max)  # Extract Log-Mel spectrogram
            feature = normalize(feature, axis=1)

        elif mode == 'augmented':
            features = []

            # Base MFCC
            base_mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sr_new, n_mfcc=52).T, axis=0)
            features.append(base_mfcc)

            # Augmented features
            for augmentation in [
                lambda d: add_noise(d, 0.001),
                lambda d: shift(d, 1600),
                lambda d: stretch(d, 1.2),
                lambda d: pitch_shift(d, sr_new, 3)
            ]:
                augmented_data = augmentation(x)
                aug_mfcc = np.mean(librosa.feature.mfcc(y=augmented_data, sr=sr_new, n_mfcc=52).T, axis=0)
                features.append(aug_mfcc)

            # Average augmented features
            feature = np.mean(features, axis=0)
            feature = normalize(feature.reshape(1, -1), axis=1).flatten()  # Normalize

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Reshape for model input if required
        if input_shape:
            feature = _reshape_feature(feature, input_shape)

        logger.info(f"Feature extracted with shape {feature.shape}.")
        return np.expand_dims(feature, axis=-1)  # Add channel dimension

    except Exception as e:
        logger.error(f"Error in preprocessing audio: {e}")
        raise


def _reshape_feature(feature, input_shape):
    """
    Reshape the feature to match the expected input shape of the model.

    Args:
        feature: The extracted feature.
        input_shape: The expected input shape of the model.

    Returns:
        Reshaped feature.
    """
    expected_time_frames = input_shape[1]
    if len(feature) > expected_time_frames:
        feature = feature[:expected_time_frames]
    elif len(feature) < expected_time_frames:
        feature = np.pad(feature, (0, expected_time_frames - len(feature)))

    return feature


def classify_audio(model_type, feature_type, file_path):
    """
    Classify an audio file using the specified model and feature type.

    Args:
        model_type: Type of model ('binary' or 'multi').
        feature_type: Feature extraction type ('mfcc', 'log_mel', or 'augmented').
        file_path: Path to the audio file.

    Returns:
        Predicted class and prediction probabilities.
    """
    try:
        model_file = os.path.join(MODEL_PATH, MODELS[model_type][feature_type])
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file '{model_file}' not found.")
        model = load_model(model_file)

        # Get input shape from the model
        input_shape = model.input_shape

        # Preprocess audio
        processed_audio = preprocess_audio(file_path, mode=feature_type, input_shape=input_shape)

        # Add batch dimension
        processed_audio = np.expand_dims(processed_audio, axis=0)

        # Predict
        predictions = model.predict(processed_audio)
        predicted_class = np.argmax(predictions, axis=1)[0]
        probabilities = predictions[0].tolist()

        logger.info(f"Prediction complete. Predicted class: {predicted_class}, Probabilities: {probabilities}")
        return predicted_class, probabilities

    except Exception as e:
        logger.error(f"Error in classification: {e}")
        raise


def main():
    logger.info("Starting audio classification test script.")

    if not os.path.exists(FILE_PATH):
        logger.error(f"Audio file not found: {FILE_PATH}")
        return

    for model_type in MODELS.keys():
        for feature_type in MODELS[model_type].keys():
            try:
                logger.info(f"Testing {model_type} model with {feature_type} features.")
                predicted_class, probabilities = classify_audio(model_type, feature_type, FILE_PATH)
                class_name = CLASS_NAMES[model_type][predicted_class]
                logger.info(f"Predicted Class: {class_name} ({predicted_class}), Probabilities: {probabilities}")
            except Exception as e:
                logger.error(f"Failed for {model_type} - {feature_type}: {e}")


if __name__ == "__main__":
    main()
