import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load pipeline components
@st.cache_resourc
def load_componets():
    scaler = joblib.load("scaler.pkl")
    selector = joblib.load("selector.pkl")
    model = joblib.load("best_model.pkl")
    return scaler, selector, model

scaler, selector, model = load_components()

# Define all original feature names (before selection) — update this!
original_features = [
    'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
    'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14'
]

st.title("HVAC Predictive Maintenance")

# Sidebar user input for all features
st.sidebar.header("Input Sensor Readings")
user_input = {}
for feature in original_features:
    user_input[feature] = st.sidebar.number_input(f"{feature}", value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Scale the input
scaled_input = scaler.transform(input_df)

# Select top 10 features
selected_input = selector.transform(scaled_input)

# Predict
prediction = model.predict(selected_input)[0]
prediction_proba = model.predict_proba(selected_input)[0]

# Optional: label mapping
label_map = {0: "Normal", 1: "Warning", 2: "Failure"}
predicted_status = label_map.get(prediction, prediction)

# Display
st.subheader("Prediction Result")
st.success(f"Predicted Machine Status: **{predicted_status}**")

st.subheader("Prediction Probabilities")
st.bar_chart(prediction_proba)
