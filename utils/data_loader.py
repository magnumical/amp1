import pandas as pd
import os
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
data_logger = logging.getLogger("data_pipeline")


def load_data():
    """Load patient diagnosis and demographic data."""
    data_logger.info("Loading patient diagnosis and demographic data.")
    diagnosis_df = pd.read_csv('D://github//AmpleHealth//data//Respiratory_Sound_Database//patient_diagnosis.csv', 
                               names=['Patient number', 'Diagnosis'])

    patient_df = pd.read_csv('D://github//AmpleHealth//data//demographic_info.txt', 
                             names=['Patient number', 'Age', 'Sex', 'Adult BMI (kg/m2)', 'Child Weight (kg)', 'Child Height (cm)'],
                             delimiter=' ')

    data_logger.info("Data successfully loaded.")
    return pd.merge(left=patient_df, right=diagnosis_df, how='left')

def process_audio_metadata(folder_path):
    """Extract audio metadata from filenames."""
    data_logger.info("Extracting audio metadata from filenames.")
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            parts = filename.split('_')
            data.append({
                'Patient number': int(parts[0]),
                'Recording index': parts[1],
                'Chest location': parts[2],
                'Acquisition mode': parts[3],
                'Recording equipment': parts[4].split('.')[0]
            })
    data_logger.info("Audio metadata extraction complete.")
    return pd.DataFrame(data)
