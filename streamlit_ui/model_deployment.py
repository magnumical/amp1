import streamlit as st
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from tensorflow.keras.models import load_model
import logging


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("audio_classifier_dep")


# Paths and Constants
MODEL_PATH = "./models"
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



## Augmentation Functions
def add_noise(data, noise_factor=0.001):
    noise = np.random.randn(len(data))
    return data + noise_factor * noise

def shift(data, shift_factor=1600):
    return np.roll(data, shift_factor)

def stretch(data, rate=1.2):
    return librosa.effects.time_stretch(data, rate=rate)

def pitch_shift(data, sr, n_steps=3):
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=n_steps)

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
    Classify an audio file using the specified model.

    Args:
        model_type: Type of model ('binary' or 'multi').
        feature_type: Type of feature extraction ('mfcc', 'log_mel', or 'augmented').
        file_path: Path to the audio file.

    Returns:
        Predicted class and prediction probabilities.
    """
    if model_type not in MODELS or feature_type not in MODELS[model_type]:
        raise ValueError(f"Invalid combination of model type and feature type: {model_type}, {feature_type}")

    # Load the correct model based on the type and feature
    model_file = os.path.join(MODEL_PATH, MODELS[model_type][feature_type])
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")

    logger.info(f"Loading model from {model_file} for feature type '{feature_type}' and model type '{model_type}'...")
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


# Streamlit App Function
def run():
    st.header("Respiratory Sound Classifier")

    # User input: Model type and feature extraction mode
    model_type = st.selectbox("Select Model Type", ["binary", "multi"], help="Choose between binary or multi-class classification.")
    feature_type = st.selectbox("Select Feature Type", ["mfcc", "log_mel", "augmented"], help="Choose the feature extraction type.")

    # User input: Upload audio file
    uploaded_file = st.file_uploader("Upload an Audio File", type=["wav", "mp3"], help="Supported formats: WAV, MP3")

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = os.path.join("temp_audio", uploaded_file.name)
        os.makedirs("temp_audio", exist_ok=True)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.audio(temp_file_path, format="audio/wav", start_time=0)


        try:
            # Perform classification
            with st.spinner("Classifying the audio file, please wait..."):
                predicted_class, probabilities = classify_audio(model_type, feature_type, temp_file_path)

            # Map class index to label
            class_label = CLASS_NAMES[model_type][predicted_class]

            # Display results
            st.success("Classification Complete!")
            st.write(f"Predicted Class: {class_label} (Index: {predicted_class})")
            st.write("Prediction Probabilities:")
            st.json({CLASS_NAMES[model_type][i]: prob for i, prob in enumerate(probabilities)})

        except Exception as e:
            st.error(f"Error: {e}")

        # Clean up the temporary file
        os.remove(temp_file_path)

if __name__ == "__main__":
    run()