import streamlit as st

def run():
    st.title("Welcome!")
    st.subheader("Introduction")
    st.write("""
This project involves developing a machine learning solution to classify respiratory sounds into diagnostic categories using the ICBHI 2017 Challenge Dataset. 
The pipeline includes data preprocessing, feature extraction, model training, evaluation, and deployment. """)
    
    st.subheader("Introduction")
    st.write(""" ### This app provise following:
    - **Data Exploration**: Visualize and preprocess respiratory sound datasets.
    - **Model Performance**: Evaluate performance metrics for different model configurations.
    - **Model Deployment**: Upload and classify your own respiratory sound files. """)
 
    st.write(""" ### Repository:
    You can access the GitHub repository for this project here:
    [GitHub Repository](https://github.com/your-repo-link) """)

    st.write(""" ### Contact:
    Developed by Reza Amini | magnumical.ca  
    """)

    st.image("./streamlit_ui/img/deployment.png", caption="Project Overview")