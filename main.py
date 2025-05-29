import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model components
@st.cache_resource
def load_components():
    scaler = joblib.load("scaler.pkl")
    selector = joblib.load("selector.pkl")
    model = joblib.load("best_model.pkl")
    return scaler, selector, model

scaler, selector, model = load_components()

# Define all original feature names
original_features = [
    'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
    'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14'
]

# Load local CSV
@st.cache_data
def load_dataset():
    df = pd.read_csv("sensor.csv")
    return df[original_features]

sensor_data = load_dataset()

st.title("🔧 HVAC Predictive Maintenance App")

# Show dataset preview
st.subheader("📊 Preview of Sensor Dataset (sensor.csv)")
st.dataframe(sensor_data.head())

# Select row to predict
row_index = st.number_input("🔢 Select Row Index for Prediction", min_value=0, max_value=len(sensor_data)-1, value=0)
input_df = sensor_data.iloc[[row_index]]

# Predict on selected row
if st.button("🔍 Predict Machine Status"):
    try:
        # Select and scale features
        selected_input = selector.transform(input_df)
        scaled_input = scaler.transform(selected_input)

        # Predict
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[0]

        label_map = {0: "Normal", 1: "Warning", 2: "Failure"}
        predicted_status = label_map.get(prediction, prediction)

        # Display results
        st.subheader("✅ Prediction Result")
        st.success(f"Predicted Machine Status: **{predicted_status}**")

        st.subheader("📊 Prediction Probabilities")
        st.bar_chart(prediction_proba)

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
