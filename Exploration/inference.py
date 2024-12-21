import os
import wave
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy.signal as signal

class RespiratorySoundAnalysis:
    def __init__(self, diagnosis_file, audio_path):
        self.diagnosis_file = diagnosis_file
        self.audio_path = audio_path
        self.diagnosis_df = None
        self.audio_files = None
        self.audio_df = None
        self.merged_df = None

    def load_diagnosis_data(self):
        """Load patient diagnosis data."""
        self.diagnosis_df = pd.read_csv(self.diagnosis_file, names=['patient_id', 'disease'])
        print("Diagnosis Data Preview:")
        print(self.diagnosis_df.head())
        
        print("\nDisease Distribution:")
        print(self.diagnosis_df['disease'].value_counts())
        
        print("\nDisease Proportion:")
        print(self.diagnosis_df['disease'].value_counts(normalize=True) * 100)

    def plot_disease_distribution(self):
        """Plot disease distribution."""
        plt.figure(figsize=(10, 6))
        sns.countplot(y=self.diagnosis_df['disease'], order=self.diagnosis_df['disease'].value_counts().index, palette='viridis')
        plt.title("Disease Distribution")
        plt.xlabel("Count")
        plt.ylabel("Disease")
        plt.show()


    def load_audio_files(self):
        """Load audio file paths."""
        print(f"Searching for .wav files in: {self.audio_path}")
        try:
            # Recursively look for .wav files
            self.audio_files = [
                os.path.join(root, file)
                for root, _, files in os.walk(self.audio_path)  # Walk through subdirectories
                for file in files
                if file.lower().endswith('.wav')  # Case-insensitive filtering
            ]
            print(f"Total audio files found: {len(self.audio_files)}")
            if not self.audio_files:
                print("No .wav files found. Check the directory or file extensions.")
        except Exception as e:
            print(f"Error while loading audio files: {e}")
            self.audio_files = []

    def extract_audio_properties(self, file_path):
        """Extract properties of an audio file."""
        with wave.open(file_path, 'r') as audio_file:
            params = audio_file.getparams()
            return {
                "n_channels": params.nchannels,
                "sample_width": params.sampwidth,
                "frame_rate": params.framerate,
                "n_frames": params.nframes,
                "duration_sec": params.nframes / params.framerate
            }

    def analyze_audio_properties(self):
        """Analyze properties of audio files."""
        if not self.audio_files:
            print("No audio files found to analyze.")
            self.audio_df = None
            return

        audio_properties = []
        for file in self.audio_files[:]:
            file_path = os.path.join(self.audio_path, file)
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
            props = self.extract_audio_properties(file_path)
            props['file_name'] = file
            audio_properties.append(props)

        if audio_properties:
            self.audio_df = pd.DataFrame(audio_properties)
            print("\nAudio File Properties:")
            print(self.audio_df.describe())
        else:
            print("No audio properties could be extracted.")
            self.audio_df = None



    def plot_audio_duration_distribution(self):
        """Plot distribution of audio durations."""
        plt.figure(figsize=(10, 6))
        sns.histplot(self.audio_df['duration_sec'], kde=True, bins=20, color='skyblue')
        plt.title("Audio Duration Distribution")
        plt.xlabel("Duration (seconds)")
        plt.ylabel("Frequency")
        plt.show()

    def visualize_sample_audio(self, file_name):
        """Visualize a sample audio file."""
        file_path = os.path.join(self.audio_path, file_name)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return

        try:
            y, sr = librosa.load(file_path)

            # Plot waveform
            plt.figure(figsize=(12, 6))
            librosa.display.waveshow(y, sr=sr)
            plt.title(f"Waveform for {file_name}")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.show()

            # Plot spectrogram
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            plt.figure(figsize=(12, 6))
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='viridis')
            plt.colorbar(format='%+2.0f dB')
            plt.title("Spectrogram")
            plt.show()
        except Exception as e:
            print(f"Error visualizing {file_name}: {e}")


    def merge_audio_and_diagnosis_data(self):
        """Combine audio stats with diagnosis data."""
        # Extract file name without the full path
        self.audio_df['file_name_only'] = self.audio_df['file_name'].apply(os.path.basename)

        # Split the file name to extract patient ID (assuming it is the first part of the file name)
        try:
            self.audio_df['patient_id'] = self.audio_df['file_name_only'].str.split('_').str[0].astype(int)
        except ValueError as e:
            print(f"Error extracting patient_id: {e}")
            print(self.audio_df['file_name_only'].head())  # Debugging information
            raise

        # Merge with diagnosis data
        self.merged_df = pd.merge(
            self.audio_df, 
            self.diagnosis_df, 
            left_on='patient_id', 
            right_on='patient_id', 
            how='left'
        )
        print("\nMerged Audio and Diagnosis Data:")
        print(self.merged_df.head())

    def preprocess_audio(self, y, sr, target_sr=7000, low_cutoff=80, high_cutoff=3000, gamma=30):
        """Preprocess audio by resampling, bandpass filtering, log compression, and normalization."""
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sos = signal.butter(10, [low_cutoff, high_cutoff], btype='bandpass', fs=target_sr, output='sos')
        y_filtered = signal.sosfilt(sos, y_resampled)
        y_compressed = np.sign(y_filtered) * np.log1p(gamma * np.abs(y_filtered)) / np.log1p(gamma)
        y_normalized = y_compressed / np.max(np.abs(y_compressed))
        return y_normalized, target_sr

# Entry point for standalone execution
if __name__ == "__main__":
    diagnosis_file = 'D://github//AmpleHealth//data//Respiratory_Sound_Database//Respiratory_Sound_Database/patient_diagnosis.csv'
    audio_path = 'D://github//AmpleHealth/data/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/'

    analysis = RespiratorySoundAnalysis(diagnosis_file, audio_path)

    # Load and analyze data
    analysis.load_diagnosis_data()
    analysis.plot_disease_distribution()
    analysis.load_audio_files()
    analysis.analyze_audio_properties()
    analysis.plot_audio_duration_distribution()

    # Visualize sample audio
    if analysis.audio_files:
        analysis.visualize_sample_audio(analysis.audio_files[0])

    # Merge data
    analysis.merge_audio_and_diagnosis_data()
