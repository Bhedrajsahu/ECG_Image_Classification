import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# =========================
# MODEL CONFIG
# =========================
MODEL_URL = "https://drive.google.com/uc?id=1_B1o8zPaOBrpr8vrRc591xv7KgT33IV4"
MODEL_PATH = "ecg_model_final_fixed.h5"

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.write("⬇️ Downloading model...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    size = os.path.getsize(MODEL_PATH)
    st.write(f"Model size: {size/1024/1024:.2f} MB")

    if size < 5 * 1024 * 1024:
        st.error("❌ Model download failed")
        st.stop()

    st.write("✅ Loading model...")
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# =========================
# CLASS LABELS
# =========================
class_names = [
    "Abnormal Heartbeat",
    "COVID-19",
    "MI",
    "MI History",
    "Normal"
]

# =========================
# UI
# =========================
st.title("ECG Image Classification")

file = st.file_uploader("Upload ECG Image", type=["jpg","png","jpeg"])

if file is not None:
    img = Image.open(file)
    st.image(img, caption="Uploaded ECG")

    img = img.resize((128,128))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1,128,128,3)

    pred = model.predict(img_array)
    cls = np.argmax(pred)
    conf = np.max(pred)

    st.success(f"Prediction: {class_names[cls]}")
    st.write(f"Confidence: {conf:.2f}")
