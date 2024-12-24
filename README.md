# Overview and Envisioned Workflow
This project focuses on developing a pipeline to classify respiratory sounds into diagnostic categories using the ICBHI 2017 dataset. The pipeline involves data preprocessing, feature extraction, model training, and evaluation. It supports binary and multi-class classification tasks. 
Please check out the instructions ([PDF](https://github.com/magnumical/amp1/blob/main/src/AmpH_Report.pdf)) or deployed web-app ([Hugging Face](https://huggingface.co/spaces/magnumical/amp)) for more descriptive info.

* Since there was only a short amount of time, I tried to mimic how I would approach such a classification problem. Therefore, I created a workflow that tests different options, from data processing to model architecture, to determine the best-performing model.

* You can also checkout ([gitHub actions]([https://huggingface.co/spaces/magnumical/amp](https://github.com/magnumical/amp1/actions))) to view a limited implementation of CI/CD pipeline.

<img src="https://github.com/magnumical/amp1/blob/main/src/deployment.png" alt="Deployment Workflow" width="65%">

## Environment Setup Instructions

1. Create a Virtual Environment
```
conda create -n myenv
conda activate myenv
```
2. Install Dependencies
```
pip install -r requirements.txt
```

## Data Preparation
Download the ICBHI 2017 dataset and put it inside ```data`` folder in the main directory. If you are using Kaggle, you can add [this](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database) dataset into your space easily.

```
|-data/
|-----demographic_info.txt
|-----Respiratory_Sound_Database/
|---------patient_diagnosis.csv
|---------filename_format.txt
|---------audio_and_txt_files/
|-------------*.wav
|-------------*.txt
```
Based on my data exploration, the dataset is highly imbalanced. Some categories (e.g., Asthma) have a handful of samples. Therefore, I aimed to do binary (Normal vs Abnormal) as well as multiclass (broader conditions like Normal, Chronic Respiratory Diseases, and Respiratory Infections).

1. Running dataset exploration code:
``` python Exploration/inference.py --diagnosis_file ./data//Respiratory_Sound_Database//patient_diagnosis.csv --audio_path ./data/Respiratory_Sound_Database/testsample```

## Training model
As I earlier said, I tried to explore different options in model/system design of a such a project. Therefore, ```Train.py``` script in the main directory contain different options that you can choose from.
For example, you can do ```binary``` or ```multiclass``` classifications using 3 different types of input: 1. MFCC, 2. Log-Mel Spectrum, 3. MFCC with augmented features.
* Note: you can skip this step since trained models are already in ```./models/``` directory.
#### Model Training Workflow

The training process involves:

* Data preprocessing (filtering, sampling, and feature extraction).
* Splitting the dataset into training, validation, and test subsets.
* Oversampling using SMOTE to balance class distribution.
* Hyperparameter optimization using Optuna.
* Experiment tracking with MLflow.
* Model saving and evaluation.

Using the command line:
```
python Train.py --metadata_path data/Respiratory_Sound_Database/audio_and_txt_files --audio_files_path data/Respiratory_Sound_Database/audio_and_txt_filesv --demographic_path data/demographic_info.tx --diagnosis_path --diagnosis_path data/Respiratory_Sound_Database/patient_diagnosis.csv --classification_modes binary --feature_types mfcc 
```
Or for example, if you want to run all the combinations of configurations:
```
python Train.py --metadata_path data/Respiratory_Sound_Database/audio_and_txt_files --audio_files_path data/Respiratory_Sound_Database/audio_and_txt_filesv --demographic_path data/demographic_info.tx --diagnosis_path --diagnosis_path data/Respiratory_Sound_Database/patient_diagnosis.csv --classification_modes binary multi --feature_types mfcc log_mel augmented
```

Alternatively, you can run in ```debug``` mode, which uses randomly generated data to check the functionality of different parts of code.
```
python Train.py --debug
```

* Note: ```./LegacyTraining/train.py``` incorporates everything in a single file and it is suitable to be used on Kaggle/Google Notebooks.


## Outputs and Logging
1. Models are saved in .h5 format inside ```./models/```
2. Unseen testing samples (out-of-bag testing) are also stored in .npy files inside ```./processed_datasets/```. These files already preprocessed and you only need to feed it to model for testing.
3. MLflow logs are available at ```./mlruns/```

   
## Testing and Inference
1. Test samples via .npy files:

```
python TestModels.py
````

this script automatically matches .npy files with .h5 models and makes the final evaluation of model.

2. I selected audio files inside ```data\Respiratory_Sound_Database\testsample``` directory.  The ```patient_diagnosis.csv``` file inside that folder shows ground truth labels.
   a) you can run ```python Model_Inference.py```:
   * for example: 
```
Model_Inference.py ./data/Respiratory_Sound_Database/testsample/157_1b1_Al_sc_Meditron.wav
```
This file will iterate over all models and will give you general overview of different model performance.

3. Use the [Hugging Face](https://huggingface.co/spaces/magnumical/amp) app to upload your audio file and see results!


## Run UI locally + metrics collection
To run the UI locally:
```streamlit run app.py```

now if you want to access to different logs via Prometheus:
```prometheus --config.file=prometheus.yml```

1. After running successfully, you can start your Grafana UI available at ```http://localhost:3000``` 
2. From the sidebar, go to Data SourcesH (it should be under Connections)
3. Here you can add Prometheus as a data source and add panels with queries to visualize metrics from Prometheus.



   
