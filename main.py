import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your pipeline components
@st.cache_resource
def load_components():
    scaler = joblib.load("scaler.pkl")
    selector = joblib.load("selector.pkl")
    model = joblib.load("best_model.pkl")
    return scaler, selector, model

scaler, selector, model = load_components()

# List of all original feature names
original_features = [
    'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
    'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14'
]

st.title("🔧 HVAC Predictive Maintenance")

# Load local CSV
@st.cache_data
def load_dataset():
    df = pd.read_csv("sensor.csv")
    return df
sensor_data = load_dataset()

st.title("🔧 HVAC Predictive Maintenance App")

# Show dataset preview
st.subheader("📊 Preview of Sensor Dataset (sensor.csv)")
st.dataframe(sensor_data.head())

# Sidebar input for all features
st.sidebar.header("🧪 Input Sensor Readings")
user_input = {}
for feature in original_features:
    user_input[feature] = st.sidebar.number_input(f"{feature}", value=0.0)

# DataFrame from user input
input_df = pd.DataFrame([user_input])

# ✅ Apply selector first, then scaler
selected_input = selector.transform(input_df)
scaled_input = scaler.transform(selected_input)

# Predict
if st.button("🔍 Predict Machine Status"):
    prediction = model.predict(scaled_input)[0]
    prediction_proba = model.predict_proba(scaled_input)[0]

    label_map = {0: "Normal", 1: "Warning", 2: "Failure"}
    predicted_status = label_map.get(prediction, prediction)

    # Output
    st.subheader("✅ Prediction Result")
    st.success(f"Predicted Machine Status: **{predicted_status}**")

    st.subheader("📊 Prediction Probabilities")
    st.bar_chart(prediction_proba)
