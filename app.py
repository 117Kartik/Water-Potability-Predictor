import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. Load the exported assets ---
@st.cache_resource
def load_assets():
    model = joblib.load('best_water_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# --- 2. Set up the Streamlit UI configuration ---
st.set_page_config(page_title="Water Potability Predictor", page_icon="💧", layout="centered")

# Hide the Streamlit Deploy button and standard menu
hide_streamlit_style = """
            <style>
            .stDeployButton {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("💧 Water Potability Predictor")
st.markdown("""
This application uses a Machine Learning model (Support Vector Machine) to predict whether water is **safe for human consumption** based on 9 water quality metrics.
""")
st.divider()

# --- 3. Build the User Input Form ---
st.subheader("Enter Water Metrics:")
col1, col2 = st.columns(2)

with col1:
    ph = st.number_input("pH Level (0-14)", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
    hardness = st.number_input("Hardness (mg/L)", min_value=0.0, max_value=500.0, value=196.0, step=1.0)
    solids = st.number_input("Total Dissolved Solids (ppm)", min_value=0.0, max_value=70000.0, value=22014.0, step=100.0)
    chloramines = st.number_input("Chloramines (ppm)", min_value=0.0, max_value=15.0, value=7.1, step=0.1)
    sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, max_value=500.0, value=333.7, step=1.0)

with col2:
    conductivity = st.number_input("Conductivity (μS/cm)", min_value=0.0, max_value=1000.0, value=426.2, step=1.0)
    organic_carbon = st.number_input("Organic Carbon (ppm)", min_value=0.0, max_value=30.0, value=14.2, step=0.1)
    trihalomethanes = st.number_input("Trihalomethanes (μg/L)", min_value=0.0, max_value=150.0, value=66.3, step=0.1)
    turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, max_value=10.0, value=3.9, step=0.1)

st.divider()

# --- 4. Prediction Logic ---
if st.button("Predict Potability", use_container_width=True):
    # Gather inputs into a numpy array matching the training data structure
    user_data = np.array([[ph, hardness, solids, chloramines, sulfate, 
                           conductivity, organic_carbon, trihalomethanes, turbidity]])
    
    # Scale the input data just like we did in the notebook
    scaled_data = scaler.transform(user_data)
    
    # Predict using the loaded model
    prediction = model.predict(scaled_data)[0]
    
    # Display results nicely
    if prediction == 1:
        st.success("**The water is POTABLE (Safe to drink).**")
        st.balloons()
    else:
        st.error("**The water is NOT POTABLE (Unsafe to drink).**")
        
st.markdown("---")