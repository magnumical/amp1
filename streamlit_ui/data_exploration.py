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
        # List all .wav files in the AUDIO_PATH directory
        wav_files = [f for f in os.listdir(AUDIO_PATH) if f.endswith('.wav')]
        
        if wav_files:
            selected_file_name = st.selectbox("Select an Audio File", wav_files)

            # Construct the full path of the selected file
            file_path = os.path.join(AUDIO_PATH, selected_file_name)

            try:
                # Load raw audio
                y_raw, sr = librosa.load(file_path)
            except Exception as e:
                st.error(f"Error loading audio file: {e}")
                st.stop()

            # Preprocessing and Visualization
            try:
                y_processed, processed_sr = analysis.preprocess_audio(y_raw, sr)

                # Mel spectrogram
                mel = librosa.feature.melspectrogram(
                    y=y_processed, sr=processed_sr, n_fft=2048, hop_length=512, power=2.0
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)

                # STFT
                stft = librosa.stft(y_processed, n_fft=2048, hop_length=512)
                stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

                # Frequency Spectrum
                fft = np.abs(np.fft.rfft(y_processed))
                freqs = np.fft.rfftfreq(len(y_processed), 1 / processed_sr)

                # Zero-Crossing Rate
                zcr = librosa.feature.zero_crossing_rate(y_processed)[0]

                # RMS Energy
                rms = librosa.feature.rms(y=y_processed)[0]

                # Create subplots for visualizations
                fig, axs = plt.subplots(3, 2, figsize=(15, 12))

                # Raw waveform
                librosa.display.waveshow(y_raw, sr=sr, ax=axs[0, 0])
                axs[0, 0].set_title("Raw Waveform", fontsize=12)

                # Preprocessed waveform
                librosa.display.waveshow(y_processed, sr=processed_sr, ax=axs[0, 1])
                axs[0, 1].set_title("Preprocessed Waveform", fontsize=12)

                # Frequency spectrum
                axs[1, 0].plot(freqs, fft, color='blue')
                axs[1, 0].set_title("Frequency Spectrum", fontsize=12)
                axs[1, 0].set_xlabel("Frequency (Hz)")
                axs[1, 0].set_ylabel("Amplitude")

                # ZCR
                axs[1, 1].plot(zcr, color='green')
                axs[1, 1].set_title("Zero-Crossing Rate", fontsize=12)
                axs[1, 1].set_xlabel("Frames")
                axs[1, 1].set_ylabel("Rate")

                # RMS Energy
                axs[2, 0].plot(rms, color='red')
                axs[2, 0].set_title("RMS Energy", fontsize=12)
                axs[2, 0].set_xlabel("Frames")
                axs[2, 0].set_ylabel("RMS")

                # Mel spectrogram
                img_mel = librosa.display.specshow(
                    mel_db, sr=processed_sr, x_axis='time', y_axis='mel', ax=axs[2, 1], cmap='viridis'
                )
                axs[2, 1].set_title("Mel Spectrogram", fontsize=12)
                fig.colorbar(img_mel, ax=axs[2, 1], format="%+2.0f dB")

                # Adjust layout
                plt.tight_layout()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error during audio preprocessing or visualization: {e}")
                st.stop()

            # Play audio
            st.subheader("Listen to Audio")
            st.audio(file_path, format="audio/wav")
        else:
            st.warning("No audio files found in the directory.")
