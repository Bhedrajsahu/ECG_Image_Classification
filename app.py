import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests

MODEL_URL = "https://drive.google.com/uc?id=1Bp7dfpa6qG6HZPuB96hGG60ZMAxO0R7K"
MODEL_PATH = "ecg_model.keras"

# Download model (only once)
@st.cache_resource
def load_model():
    import os
    if not os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "wb") as f:
            response = requests.get(MODEL_URL)
            f.write(response.content)

    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

class_names = [
    "Abnormal Heartbeat",
    "COVID-19",
    "MI",
    "MI History",
    "Normal"
]

st.title("ECG Classification")

file = st.file_uploader("Upload ECG Image", type=["jpg","png","jpeg"])

if file is not None:
    img = Image.open(file)
    st.image(img, caption="Uploaded ECG", use_column_width=True)

    img = img.resize((128,128))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1,128,128,3)

    pred = model.predict(img_array)
    cls = np.argmax(pred)
    conf = np.max(pred)

    st.success(f"Prediction: {class_names[cls]}")
    st.write(f"Confidence: {conf:.2f}")
