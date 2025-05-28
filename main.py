import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI()

# Define input format
class SensorInput(BaseModel):
    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float

@app.post("/predict")
def predict(data: SensorInput):
    input_data = np.array([[  # match your feature order
        data.sensor_1, data.sensor_2, data.sensor_3, data.sensor_4, data.sensor_5,
        data.sensor_6, data.sensor_7, data.sensor_8, data.sensor_9, data.sensor_10
    ]])

    scaled = scaler.transform(input_data)
    prediction = model.predict(scaled)

    return {"prediction": int(prediction[0])}
