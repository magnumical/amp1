import streamlit as st
from streamlit_ui import readme, data_exploration, model_performance, model_deployment

# Load custom CSS
with open("streamlit_ui/style.css") as css_file:
    st.markdown(f'<style>{css_file.read()}</style>', unsafe_allow_html=True)

# Initialize session state for the selected page
if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Introduction"

# Streamlit app setup
st.title("ICBHI 2017 Challenge - Amplifier Health")

# Sidebar Navigation
st.sidebar.markdown('<div class="sidebar-header">Navigate</div>', unsafe_allow_html=True)

if st.sidebar.button("Introduction"):
    st.session_state["active_page"] = "Introduction"
if st.sidebar.button("Data Exploration"):
    st.session_state["active_page"] = "Data Exploration"
if st.sidebar.button("Model Performance"):
    st.session_state["active_page"] = "Model Performance"
if st.sidebar.button("Model Deployment"):
    st.session_state["active_page"] = "Model Deployment"

# Page Content
if st.session_state["active_page"] == "Introduction":
    readme.run()
elif st.session_state["active_page"] == "Data Exploration":
    data_exploration.run()
elif st.session_state["active_page"] == "Model Performance":
    model_performance.run()
elif st.session_state["active_page"] == "Model Deployment":
    model_deployment.run()
