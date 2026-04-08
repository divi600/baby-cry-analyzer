import os

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np

from utils import extract_features, le

# ✅ CREATE APP FIRST
app = FastAPI()

# ✅ CORS (important for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ LOAD MODEL
model = tf.keras.models.load_model("model.keras")

# ✅ API ENDPOINT
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    print("🔥 Audio received:", file.filename)

    features = extract_features(audio_bytes)
    prediction = model.predict(features)

    idx = np.argmax(prediction)
    label = le.inverse_transform([idx])[0]
    confidence = float(np.max(prediction) * 100)

    return {
        "prediction": label,
        "confidence": confidence
    }
