import streamlit as st
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from tensorflow.keras.models import load_model
import logging
from prometheus_client import Counter, Histogram, start_http_server
import time
from scipy.signal import butter, sosfilt
import pandas as pd
 

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

# Define Prometheus metrics
REQUEST_COUNT = Counter('audio_classifier_requests_total', 'Total number of requests to the classifier')
RESPONSE_TIME = Histogram('audio_classifier_response_time_seconds', 'Time taken to process requests')
ERROR_COUNT = Counter('audio_classifier_errors_total', 'Total number of errors during classification')

# Start Prometheus HTTP server
start_http_server(9100, addr="0.0.0.0")  # Expose metrics on 9100 for external scraping
# Expose metrics at http://localhost:9100/metrics


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


def classify_audio_with_metrics(model_type, feature_type, file_path):
    """
    Wrapper around classify_audio to include Prometheus metrics.
    """
    REQUEST_COUNT.inc()  # Increment request counter
    start_time = time.time()
    try:
        # Call the original classify_audio function
        result = classify_audio(model_type, feature_type, file_path)
        return result
    except Exception as e:
        ERROR_COUNT.inc()  # Increment error counter
        raise
    finally:
        RESPONSE_TIME.observe(time.time() - start_time)  # Observe response time

def run():
    st.title("Respiratory Sound Classifier: Inference and Deployment")

    st.markdown("""
    Welcome to the **Inference and Deployment** page! This tool allows you to classify respiratory sounds 
    into various categories using pre-trained models. Choose one of the two modes below based on your needs:

    - **Quick Multiclass Mode:** A fast and straightforward way to classify audio files using a multiclass model with augmented features.
    - **Flexible Mode:** Customize the classification process by selecting your preferred model type (binary/multi) and feature type (MFCC, Log-Mel, or Augmented).
    - **Metrics Dashboard:** Monitor live metrics including request counts, response times, and error rates.
    """)

    # Tabs for three modes
    tab1, tab2, tab3 = st.tabs(["Quick Multiclass Mode", "Flexible Mode", "Metrics Dashboard"])

    # Tab 1: Quick Multiclass (Augmented) Mode
    with tab1:
        st.subheader("Quick Multiclass (Augmented) Mode")
        st.markdown("""
        This mode is optimized for quick classification of respiratory sounds into multiple categories 
        (e.g., Chronic Respiratory Diseases, Normal, Respiratory Infections). It automatically uses the 
        multiclass model with augmented features for robust and accurate results.
        """)
        
        uploaded_file = st.file_uploader(
            "Upload an Audio File for Multiclass Classification",
            type=["wav", "mp3"],
            help="Supported formats: WAV, MP3",
        )

        if uploaded_file is not None:
            temp_file_path = save_uploaded_file(uploaded_file)
            st.audio(temp_file_path, format="audio/wav", start_time=0)

            try:
                with st.spinner("Classifying the audio file, please wait..."):
                    predicted_class, probabilities = classify_audio_with_metrics(
                        model_type="multi", feature_type="augmented", file_path=temp_file_path
                    )

                # Display results
                display_results(predicted_class, probabilities, "multi")

            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                os.remove(temp_file_path)

    # Tab 2: Flexible Mode
    with tab2:
        st.subheader("Flexible Mode")
        st.markdown("""
        The Flexible Mode gives you control over the classification process. Select the model type 
        (binary or multiclass) and the feature type (MFCC, Log-Mel, or Augmented) to suit your specific requirements.
        """)
        
        model_type = st.selectbox(
            "Select Model Type",
            ["binary", "multi"],
            help="Choose between binary or multi-class classification.",
        )
        feature_type = st.selectbox(
            "Select Feature Type",
            ["mfcc", "log_mel", "augmented"],
            help="Choose the feature extraction type.",
        )
        uploaded_file = st.file_uploader(
            "Upload an Audio File",
            type=["wav", "mp3"],
            help="Supported formats: WAV, MP3",
        )

        if uploaded_file is not None:
            temp_file_path = save_uploaded_file(uploaded_file)
            st.audio(temp_file_path, format="audio/wav", start_time=0)

            try:
                with st.spinner("Classifying the audio file, please wait..."):
                    predicted_class, probabilities = classify_audio_with_metrics(
                        model_type, feature_type, temp_file_path
                    )

                # Display results
                display_results(predicted_class, probabilities, model_type)

            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                os.remove(temp_file_path)

    # Tab 3: Metrics Dashboard
    # Tab 3: Metrics Dashboard


    with tab3:
        st.subheader("Metrics Dashboard")
        st.markdown("""
        This dashboard shows live metrics for the application, including request counts, response times, 
        and error counts. These metrics are tracked internally and updated in real-time.
        """)

        # Real-time metrics visualization
        col1, col2, col3 = st.columns(3)

        # Display live metrics
        with col1:
            st.metric("Total Requests", REQUEST_COUNT._value.get())
        with col2:
            st.metric("Total Errors", ERROR_COUNT._value.get())
        with col3:
            # Calculate average response time
            response_time_sum = RESPONSE_TIME._sum.get() if hasattr(RESPONSE_TIME, '_sum') else 0
            response_time_count = RESPONSE_TIME._count.get() if hasattr(RESPONSE_TIME, '_count') else 0
            avg_response_time = response_time_sum / response_time_count if response_time_count > 0 else 0
            st.metric("Avg Response Time (s)", f"{avg_response_time:.3f}")

        # Response Time Histogram Visualization
        st.markdown("### Response Time Distribution")

        if hasattr(RESPONSE_TIME, "_buckets"):
            # Exclude the +Inf bucket
            response_time_data = RESPONSE_TIME._buckets[:-1]  
            response_time_labels = [f"<= {bucket}" for bucket in range(1, len(response_time_data)+1)]

            # Create a DataFrame for bucket counts
            response_time_df = pd.DataFrame({
                "Time Range": response_time_labels,
                "Request Count": response_time_data
            })

            # Set the time range as the index
            response_time_df.set_index("Time Range", inplace=True)

            # Display the bar chart
            st.bar_chart(response_time_df)

            # Show the response time sum and average if the data is available
            st.markdown(f"**Total Response Time Sum**: {response_time_sum:.3f} seconds")
            st.markdown(f"**Average Response Time**: {avg_response_time:.3f} seconds")

        else:
            st.warning("No response time data available for visualization.")

def save_uploaded_file(uploaded_file):
    """Save the uploaded file temporarily."""
    temp_file_path = os.path.join("temp_audio", uploaded_file.name)
    os.makedirs("temp_audio", exist_ok=True)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_file_path


def display_results(predicted_class, probabilities, model_type):
    """Display the classification results."""
    class_label = CLASS_NAMES[model_type][predicted_class]
    st.success(f"Classification Complete! Predicted Class: **{class_label}**")
    st.write("### Prediction Probabilities")
    class_probabilities = {
        CLASS_NAMES[model_type][i]: prob for i, prob in enumerate(probabilities)
    }
    st.bar_chart(class_probabilities)


if __name__ == "__main__":
    run()
