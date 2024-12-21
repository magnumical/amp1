import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from Exploration.inference import RespiratorySoundAnalysis
import seaborn as sns
import os
# Initialize analysis object

DIAGNOSIS_FILE = 'D://github//AmpleHealth//data//Respiratory_Sound_Database//Respiratory_Sound_Database/patient_diagnosis.csv'
AUDIO_PATH = 'D://github//AmpleHealth/data/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/'

analysis = RespiratorySoundAnalysis(DIAGNOSIS_FILE, AUDIO_PATH)

# Load data
@st.cache_data
def load_data():
    analysis.load_diagnosis_data()
    analysis.load_audio_files()
    analysis.analyze_audio_properties()
    return analysis.diagnosis_df, analysis.audio_files, analysis.audio_df


diagnosis_df, audio_files, audio_df = load_data()

# Streamlit App
st.title("Respiratory Sound Data Explorer")

# Sidebar
st.sidebar.title("Navigation")
exploration_tab = st.sidebar.radio("Select a Tab:", ["Overview", "Explore Data", "Preprocessing & Audio Effects"])

if exploration_tab == "Overview":
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
    sns.barplot(y=disease_counts.index, x=disease_counts.values, palette="viridis", ax=ax,legend=False)
    ax.set_title("Disease Distribution", fontsize=16, fontweight='bold')
    ax.set_xlabel("Number of Patients", fontsize=12)
    ax.set_ylabel("Disease", fontsize=12)
    for i, v in enumerate(disease_counts.values):
        ax.text(v + 1, i, str(v), color='black', fontsize=10, va='center')
    st.pyplot(fig)

    # Proportion of Diseases
    st.subheader("Disease Proportion")
    disease_proportions = diagnosis_df['disease'].value_counts(normalize=True) * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(y=disease_proportions.index, x=disease_proportions.values, palette="coolwarm", ax=ax,legend=False)
    ax.set_title("Disease Proportion (%)", fontsize=16, fontweight='bold')
    ax.set_xlabel("Proportion (%)", fontsize=12)
    ax.set_ylabel("Disease", fontsize=12)
    for i, v in enumerate(disease_proportions.values):
        ax.text(v + 0.5, i, f"{v:.1f}%", color='black', fontsize=10, va='center')
    st.pyplot(fig)


if exploration_tab == "Explore Data":
    st.header("Explore Data")

    if audio_df is not None and not audio_df.empty:
        # Key Audio Insights
        total_files = len(audio_df)
        avg_duration = audio_df['duration_sec'].mean()
        min_duration = audio_df['duration_sec'].min()
        max_duration = audio_df['duration_sec'].max()
        shortest_file = audio_df.loc[audio_df['duration_sec'].idxmin(), 'file_name']
        longest_file = audio_df.loc[audio_df['duration_sec'].idxmax(), 'file_name']

        st.subheader("Key Audio Insights")
        st.markdown(f"""
        - **Total Audio Files:** {total_files}
        - **Average Duration:** {avg_duration:.2f} seconds
        - **Shortest Audio File:** {shortest_file} ({min_duration:.2f} seconds)
        - **Longest Audio File:** {longest_file} ({max_duration:.2f} seconds)
        """)

        # Duration Distribution
        st.subheader("Audio Duration Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(audio_df['duration_sec'], bins=20, kde=True, color='skyblue', ax=ax)
        ax.set_title("Audio Duration Distribution", fontsize=16, fontweight='bold')
        ax.set_xlabel("Duration (seconds)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        st.pyplot(fig)

        # Highlight Outliers
        st.subheader("Audio Duration Outliers")
        outlier_threshold = st.slider(
            "Set Outlier Threshold (seconds):",
            min_value=1.0,
            max_value=float(max_duration),
            value=25.0,
            step=0.5,
        )
        outliers = audio_df[audio_df['duration_sec'] > outlier_threshold]
        if not outliers.empty:
            st.markdown(f"**Number of Outliers:** {len(outliers)}")
            st.write(outliers[['file_name', 'duration_sec']])
        else:
            st.markdown("No outliers found above the threshold.")

        # Optional Filtering
        st.subheader("Filter Audio Files by Duration")
        min_range, max_range = st.slider(
            "Select Duration Range (seconds):",
            min_value=0.0,
            max_value=float(max_duration),
            value=(0.0, float(max_duration)),
            step=0.5,
        )
        filtered_files = audio_df[
            (audio_df['duration_sec'] >= min_range) & (audio_df['duration_sec'] <= max_range)
        ]
        st.write(f"**Number of Files in Range:** {len(filtered_files)}")
        st.write(filtered_files[['file_name', 'duration_sec']])
    else:
        st.warning("No audio data available to display.")


if exploration_tab == "Preprocessing & Audio Effects":
    st.header("Preprocessing & Audio Effects")

    # Select disease and corresponding audio files
    selected_disease = st.selectbox("Select a Disease", diagnosis_df['disease'].unique())

    # Get matching patient IDs
    matching_patients = diagnosis_df[diagnosis_df['disease'] == selected_disease]['patient_id']
    st.write(f"Matching patient IDs for {selected_disease}: {list(matching_patients)}")

    # Filter files by patient ID
    disease_files = [
        file for file in audio_files
        if any(str(patient_id) in os.path.basename(file) for patient_id in matching_patients)
    ]

    if disease_files:
        selected_file = st.selectbox("Select an Audio File", disease_files)

        # Load raw audio
        file_path = selected_file
        try:
            y_raw, sr = librosa.load(file_path)
        except Exception as e:
            st.error(f"Error loading audio file: {e}")
            st.stop()

        # Preprocessing
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

            # Create subplots for additional visualizations
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
        st.warning("No audio files found for the selected disease.")
