import streamlit as st
from streamlit_ui import data_exploration, model_performance, model_deployment

# Streamlit app setup
st.title("ICPR app")

# Tabs for navigation
tabs = st.tabs(["Data Exploration", "Model Performance", "Model Deployment"])

# Data Exploration Tab
with tabs[0]:
    data_exploration.run()

# Model Performance Tab
with tabs[1]:
    model_performance.run()

# Model Deployment Tab
with tabs[2]:
    model_deployment.run()

