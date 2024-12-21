import os
import numpy as np
import pandas as pd
import librosa
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LeakyReLU, Dropout, SpatialDropout2D, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import tensorflow as tf
import joblib

# Constants
gSampleRate = 7000
upperCutoffFreq = 3000
windowSizeSeconds = 0.05
windowSampleSize = int(gSampleRate * windowSizeSeconds)
fftWindowSizeSeconds = 0.025
fftWindowSizeSamples = int(fftWindowSizeSeconds * gSampleRate)
modelWindowSize = 4.0  # in seconds

# Helper Functions
def load_audio_files(file_list):
    buffers = []
    for file in file_list:
        buffer, rate = librosa.load(file, sr=None, mono=True)
        if rate != gSampleRate:
            buffer = librosa.resample(buffer, orig_sr=rate, target_sr=gSampleRate)
        buffers.append(buffer)
    return buffers

def preprocess_audio(buffers):
    high_pass = signal.firwin(401, [80, upperCutoffFreq], pass_zero='bandpass', fs=gSampleRate)
    processed = []
    for buffer in buffers:
        filtered = signal.lfilter(high_pass, 1.0, buffer)
        compressed = np.log1p(np.abs(filtered) * 30)
        normalized = filtered / np.max(np.abs(filtered))
        processed.append(normalized)
    return processed

def compute_spectrograms(buffers):
    spectrograms = []
    for buffer in buffers:
        freq, times, spectrum = signal.spectrogram(buffer, gSampleRate, nperseg=windowSampleSize)
        spectrograms.append((freq, times, spectrum))
    return spectrograms

def extract_features(spectrograms):
    features = []
    for spec in spectrograms:
        frequencies, times, spectrum = spec
        power_envelope = np.sum(spectrum * frequencies[:, None], axis=0)
        smoothed = gaussian_filter1d(power_envelope, sigma=3)
        features.append(smoothed)
    return features

def create_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(96, kernel_size=7, activation=LeakyReLU(0.1)),
        MaxPooling2D(2),
        Conv2D(128, kernel_size=5, activation=LeakyReLU(0.1)),
        MaxPooling2D(2),
        Conv2D(128, kernel_size=3, activation=LeakyReLU(0.1)),
        MaxPooling2D(2),
        SpatialDropout2D(0.1),
        Conv2D(256, kernel_size=3, activation=LeakyReLU(0.1)),
        MaxPooling2D(2),
        SpatialDropout2D(0.1),
        Flatten(),
        Dense(4096, activation=LeakyReLU(0.1)),
        Dropout(0.5),
        Dense(256, activation=LeakyReLU(0.1)),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def visualize_results(true_labels, predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(true_labels, label='True Labels', marker='o')
    plt.plot(predictions, label='Predictions', marker='x')
    plt.legend()
    plt.title("True Labels vs Predictions")
    plt.xlabel("Cycle Index")
    plt.ylabel("Label")
    plt.show()

# Paths
audio_dir = "D://github//AmpleHealth/data/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/"
output_model_path = "respiratory_model.h5"


# Load audio files and metadata
audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]
annotations = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.txt')]

# Preprocess data
audio_buffers = load_audio_files(audio_files)
processed_buffers = preprocess_audio(audio_buffers)
spectrograms = compute_spectrograms(processed_buffers)
features = extract_features(spectrograms)

# Simulate Labels (for demonstration purposes)
labels = np.random.randint(0, 2, len(features))  # Binary labels: 0 or 1
one_hot_labels = to_categorical(labels)

# Train/Test Split
train_data, test_data, train_labels, test_labels = train_test_split(features, one_hot_labels, test_size=0.2, random_state=42)


# Pad or truncate features to a fixed length
def pad_or_truncate(features, fixed_length):
    padded_features = []
    for feature in features:
        if len(feature) < fixed_length:
            # Pad with zeros
            padded = np.pad(feature, (0, fixed_length - len(feature)), mode='constant')
        else:
            # Truncate
            padded = feature[:fixed_length]
        padded_features.append(padded)
    return np.array(padded_features)

# Define a fixed length for all features
fixed_length = max(len(f) for f in features)  # You can choose a smaller value if needed

# Apply padding/truncating to train and test data
train_data = pad_or_truncate(train_data, fixed_length)
test_data = pad_or_truncate(test_data, fixed_length)

# Expand dimensions for CNN input
train_data = np.expand_dims(train_data, axis=-1)
test_data = np.expand_dims(test_data, axis=-1)


# Train Model
input_shape = (len(train_data[0]), 1)
train_data = np.expand_dims(train_data, axis=-1)
test_data = np.expand_dims(test_data, axis=-1)

model = create_model(input_shape)
history = model.fit(
    train_data,
    train_labels,
    validation_data=(test_data, test_labels),
    epochs=10,
    batch_size=32
)

# Save Model
model.save(output_model_path)

# Test Model
test_audio = processed_buffers[0]  # Test on the first audio file
test_spec = compute_spectrograms([test_audio])[0]
predicted_labels = model.predict(np.expand_dims(test_spec, axis=0))

# Visualization
true_labels = np.argmax(test_labels, axis=1)
predicted_classes = np.argmax(predicted_labels, axis=1)
visualize_results(true_labels, predicted_classes)

print(f"Model saved at {output_model_path}")
