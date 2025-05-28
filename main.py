import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("🚨 Predictive Maintenance for HVAC Pumps")

st.markdown("Enter sensor values to predict pump status:")

# Input form
with st.form("input_form"):
    sensor_1 = st.number_input("Sensor 1", value=0.0)
    sensor_2 = st.number_input("Sensor 2", value=0.0)
    sensor_3 = st.number_input("Sensor 3", value=0.0)
    sensor_4 = st.number_input("Sensor 4", value=0.0)
    sensor_5 = st.number_input("Sensor 5", value=0.0)
    sensor_6 = st.number_input("Sensor 6", value=0.0)
    sensor_7 = st.number_input("Sensor 7", value=0.0)
    sensor_8 = st.number_input("Sensor 8", value=0.0)
    sensor_9 = st.number_input("Sensor 9", value=0.0)
    sensor_10 = st.number_input("Sensor 10", value=0.0)
    
    submit = st.form_submit_button("Predict")

if submit:
    input_data = np.array([[
        sensor_1, sensor_2, sensor_3, sensor_4, sensor_5,
        sensor_6, sensor_7, sensor_8, sensor_9, sensor_10
    ]])

    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]

    label_map = {0: "NORMAL", 1: "RECOVERING", 2: "BROKEN"}
    st.success(f"🔧 Prediction: **{label_map.get(prediction, 'Unknown')}**")
