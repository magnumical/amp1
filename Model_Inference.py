import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths
MODELS_DIR = "./models"
PROCESSED_DIR = "./processed_datasets"
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

# Load test datasets
if mode == 'gru':
    X_test_path = os.path.join(PROCESSED_DIR, "X_test_gru.npy")
    y_test_path = os.path.join(PROCESSED_DIR, "y_test_gru.npy")
else:
    X_test_path = os.path.join(PROCESSED_DIR, "X_test_Binary.npy")
    y_test_path = os.path.join(PROCESSED_DIR, "y_test_Binary.npy")

print("Loading test data...")
X_test = np.load(X_test_path)
y_test = np.load(y_test_path)

print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Perform batch predictions
print("Performing batch inference...")
predictions = model.predict(X_test)  # Predict probabilities
predicted_classes = np.argmax(predictions, axis=1)  # Convert probabilities to binary class labels (0 or 1)

# Decode y_test from one-hot encoding to class labels
y_test_binary = np.argmax(y_test, axis=1)

# Evaluate model
accuracy = np.mean(predicted_classes == y_test_binary)
print(f"Model Accuracy on test set: {accuracy * 100:.2f}%")

# Display classification report and confusion matrix
cm = confusion_matrix(y_test_binary, predicted_classes)
report = classification_report(y_test_binary, predicted_classes, target_names=["Class 0", "Class 1"])

print("\nClassification Report:")
print(report)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
