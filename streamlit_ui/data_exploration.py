import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import os
from Exploration.inference import RespiratorySoundAnalysis

# Define base paths
BASE_PATH = './/data//Respiratory_Sound_Database'
DIAGNOSIS_FILE = os.path.join(BASE_PATH, 'patient_diagnosis.csv')
AUDIO_PATH = os.path.join(BASE_PATH, 'testsample')
DEMOGRAPHIC_FILE = os.path.join('.//data', 'demographic_info.txt')

# Initialize analysis object
analysis = RespiratorySoundAnalysis(DIAGNOSIS_FILE, AUDIO_PATH)

# Load data
@st.cache_data
def load_data():
    analysis.load_diagnosis_data()
    analysis.load_audio_files()
    analysis.analyze_audio_properties()
    return analysis.diagnosis_df, analysis.audio_df

diagnosis_df, audio_df = load_data()

# Load patient demographic data
@st.cache_data
def load_patient_demographics():
    patient_df = pd.read_csv(
        DEMOGRAPHIC_FILE, 
        names=['Patient number', 'Age', 'Sex', 'Adult BMI (kg/m2)', 'Child Weight (kg)', 'Child Height (cm)'],
        delimiter=' '
    )
    return patient_df

patient_df = load_patient_demographics()

# Streamlit App Function
def run():
    st.title("Respiratory Sound Data Explorer")

    # Tabs for navigation
    tabs = st.tabs(["Overview", "Explore Data", "Patient Demographics", "Preprocessing & Audio Effects"])

    # Overview Tab
    with tabs[0]:
        st.header("Dataset Overview")

        # Highlight key statistics
        total_patients = len(diagnosis_df)
        most_common_disease = diagnosis_df['disease'].value_counts().idxmax()
        least_common_disease = diagnosis_df['disease'].value_counts().idxmin()

        st.subheader("Key Statistics")
        st.markdown(f"""
        - **Total Patients:** {total_patients}
        - **Most Common Disease:** {most_common_disease} ({diagnosis_df['disease'].value_counts().max()} patients)
        - **Least Common Disease:** {least_common_disease} ({diagnosis_df['disease'].value_counts().min()} patients)
        """)

        # Diagnosis Distribution
        st.subheader("Diagnosis Distribution")
        disease_counts = diagnosis_df['disease'].value_counts()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(y=disease_counts.index, x=disease_counts.values, palette="viridis", ax=ax)
        ax.set_title("Disease Distribution", fontsize=16, fontweight='bold')
        ax.set_xlabel("Number of Patients", fontsize=12)
        ax.set_ylabel("Disease", fontsize=12)
        st.pyplot(fig)

    # Explore Data Tab
    with tabs[1]:
        st.header("Explore Data")
        shortest_file = os.path.basename(audio_df.loc[audio_df['duration_sec'].idxmin(), 'file_name'])
        longest_file = os.path.basename(audio_df.loc[audio_df['duration_sec'].idxmax(), 'file_name'])

        if audio_df is not None and not audio_df.empty:
            st.subheader("Key Audio Insights")
            st.markdown(f"""
            - **Total Audio Files:** {len(audio_df)}
            - **Average Duration:** {audio_df['duration_sec'].mean():.2f} seconds
            - **Shortest Audio File:** {shortest_file} ({audio_df['duration_sec'].min():.2f} seconds)
            - **Longest Audio File:** {longest_file} ({audio_df['duration_sec'].max():.2f} seconds)
            """)

            # Duration Distribution
            st.subheader("Audio Duration Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(audio_df['duration_sec'], bins=20, kde=True, color='skyblue', ax=ax)
            ax.set_title("Audio Duration Distribution", fontsize=16, fontweight='bold')
            st.pyplot(fig)

        else:
            st.warning("No audio data available to display.")

    # Patient Demographics Tab
    with tabs[2]:
        st.header("Patient Demographics")
        st.dataframe(patient_df)

        st.subheader("Missing Values Information")
        st.write(patient_df.isna().sum())

        st.subheader("Age Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(patient_df['Age'].dropna(), bins=20, kde=True, color='skyblue', ax=ax)
        ax.set_title("Age Distribution", fontsize=16, fontweight='bold')
        st.pyplot(fig)

    # Preprocessing & Audio Effects Tab
    with tabs[3]:
        st.header("Preprocessing & Audio Effects")
        wav_files = [f for f in os.listdir(AUDIO_PATH) if f.endswith('.wav')]

        if wav_files:
            selected_file_name = st.selectbox("Select an Audio File", wav_files)
            file_path = os.path.join(AUDIO_PATH, selected_file_name)

            try:
                y_raw, sr = librosa.load(file_path)
                st.audio(file_path, format="audio/wav")

                # Raw Waveform
                st.subheader("Raw Waveform")
                fig, ax = plt.subplots(figsize=(10, 4))
                librosa.display.waveshow(y_raw, sr=sr, ax=ax)
                ax.set_title("Raw Waveform", fontsize=16, fontweight='bold')
                st.pyplot(fig)

                # Noise Filtering
                st.subheader("Noise Filtering")
                y_filtered = librosa.effects.preemphasis(y_raw)
                fig, ax = plt.subplots(figsize=(10, 4))
                librosa.display.waveshow(y_filtered, sr=sr, ax=ax)
                ax.set_title("Filtered Waveform (Pre-emphasis Applied)", fontsize=16, fontweight='bold')
                st.pyplot(fig)

                # Log Mel-Spectrogram
                st.subheader("Log Mel-Spectrogram")
                mel_spect = librosa.feature.melspectrogram(y=y_filtered, sr=sr, n_mels=128)
                mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
                fig, ax = plt.subplots(figsize=(10, 6))
                img = librosa.display.specshow(mel_spect_db, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='viridis')
                fig.colorbar(img, ax=ax, format="%+2.0f dB")
                ax.set_title("Log Mel-Spectrogram", fontsize=16, fontweight='bold')
                st.pyplot(fig)

                # Fast Fourier Transform (FFT)
                st.subheader("FFT (Frequency Domain)")
                fft_vals = np.abs(np.fft.fft(y_filtered))
                freqs = np.fft.fftfreq(len(fft_vals), 1 / sr)
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(freqs[:len(freqs) // 2], fft_vals[:len(fft_vals) // 2], color='blue')
                ax.set_title("FFT - Frequency Spectrum", fontsize=16, fontweight='bold')
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Amplitude")
                st.pyplot(fig)

                # Zero Crossing Rate
                st.subheader("Zero Crossing Rate")
                zcr = librosa.feature.zero_crossing_rate(y_filtered)[0]
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(zcr, color='orange')
                ax.set_title("Zero Crossing Rate Over Time", fontsize=16, fontweight='bold')
                ax.set_xlabel("Frame Index")
                ax.set_ylabel("Zero Crossing Rate")
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error processing audio file: {e}")

        else:
            st.warning("No audio files found in the directory.")

