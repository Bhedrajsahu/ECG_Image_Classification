import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# =========================
# 🔹 MODEL CONFIG
# =========================
MODEL_URL = "https://drive.google.com/uc?id=1Bp7dfpa6qG6HZPuB96hGG60ZMAxO0R7K"
MODEL_PATH = "ecg_model.keras"

# =========================
# 🔹 LOAD MODEL (CACHED)
# =========================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model... please wait ⏳")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# =========================
# 🔹 CLASS LABELS
# =========================
class_names = [
    "Abnormal Heartbeat",
    "COVID-19",
    "MI",
    "MI History",
    "Normal"
]

# =========================
# 🔹 UI
# =========================
st.title("ECG Image Classification")

uploaded_file = st.file_uploader("Upload ECG Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded ECG", use_column_width=True)

    # =========================
    # 🔹 PREPROCESSING
    # =========================
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 128, 128, 3)

    # =========================
    # 🔹 PREDICTION
    # =========================
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"Prediction: {class_names[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}")
