import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical, normalize
import matplotlib.pyplot as plt

# Define paths
MODELS_DIR = "./models"
MODEL_GRU = os.path.join(MODELS_DIR, "best_gru_model.keras")
MODEL_LOG_MEL = os.path.join(MODELS_DIR, "best_model_log_mel.h5")
MODEL_MFCC = os.path.join(MODELS_DIR, "best_model_mfcc.h5")

# Select model
print("Available Models:")
print("1: GRU")
print("2: Log Mel CNN")
print("3: MFCC CNN")
choice = input("Enter the model number you want to use for inference (1/2/3): ")

if choice == '1':
    model_path = MODEL_GRU
    mode = 'gru'
elif choice == '2':
    model_path = MODEL_LOG_MEL
    mode = 'log_mel'
elif choice == '3':
    model_path = MODEL_MFCC
    mode = 'mfcc'
else:
    raise ValueError("Invalid choice. Please select 1, 2, or 3.")

print(f"Loading model from {model_path}...")
model = load_model(model_path)
print("Model loaded successfully!")

# Define preprocessing function
def preprocess_audio(audio_file, mode):
    """
    Preprocess audio file for inference based on the selected mode.

    Args:
        audio_file: Path to the audio file.
        mode: Feature extraction mode ('gru', 'log_mel', 'mfcc').

    Returns:
        Preprocessed feature ready for inference.
    """
    sr_new = 16000  # Resample audio to 16 kHz
    x, sr = librosa.load(audio_file, sr=sr_new)

    # Padding or truncating to 5 seconds (5 * sr_new samples)
    max_len = 5 * sr_new
    if x.shape[0] < max_len:
        x = np.pad(x, (0, max_len - x.shape[0]))
    else:
        x = x[:max_len]

    if mode == 'mfcc':
        feature = librosa.feature.mfcc(y=x, sr=sr_new, n_mfcc=20)  # 20 MFCCs
    elif mode == 'log_mel':
        feature = librosa.feature.melspectrogram(y=x, sr=sr_new, n_mels=20, fmax=8000)
        feature = librosa.power_to_db(feature, ref=np.max)
    elif mode == 'gru':
        feature = librosa.feature.mfcc(y=x, sr=sr_new, n_mfcc=52)  # Match GRU feature extraction
    else:
        raise ValueError("Invalid mode. Must be one of 'mfcc', 'log_mel', or 'gru'.")

    feature = np.expand_dims(feature, axis=-1)  # Add channel dimension
    feature = normalize(feature, axis=1)  # Normalize features

    if mode == 'gru':
        feature = np.expand_dims(feature, axis=0)  # GRU expects an additional dimension
    return feature

# Perform inference
def perform_inference(audio_file):
    """
    Perform inference on a single audio file.

    Args:
        audio_file: Path to the audio file.

    Returns:
        Predicted class and probabilities.
    """
    print(f"Processing {audio_file}...")
    feature = preprocess_audio(audio_file, mode)
    predictions = model.predict(np.expand_dims(feature, axis=0))  # Expand batch dimension
    predicted_class = np.argmax(predictions, axis=1)
    probabilities = predictions[0]
    return predicted_class, probabilities

# Test the script
if __name__ == "__main__":
    # Provide the path to a test audio file
    test_audio_file = input("Enter the path to the audio file for inference: ")

    if not os.path.exists(test_audio_file):
        raise FileNotFoundError(f"The file {test_audio_file} does not exist!")

    # Perform inference
    predicted_class, probabilities = perform_inference(test_audio_file)

    # Display results
    print("\nInference Results:")
    print(f"Predicted Class: {predicted_class}")
    print(f"Class Probabilities: {probabilities}")

    # Optional: Plot the probabilities
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(probabilities)), probabilities)
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.title("Class Probabilities")
    plt.show()
