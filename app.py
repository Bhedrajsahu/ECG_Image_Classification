import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# =========================
# MODEL (WEIGHTS ONLY)
# =========================
MODEL_URL = "https://drive.google.com/uc?id=1LgYD6_BaTo4rs7eGuLuWDae_WRD0Czof"
MODEL_PATH = "ecg_model.weights.h5"

# =========================
# BUILD MODEL
# =========================
def build_model():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Flatten(),

        Dense(256, activation='relu'),
        Dropout(0.5),

        Dense(128, activation='relu'),
        Dropout(0.5),

        Dense(5, activation='softmax')
    ])
    return model

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.write("⬇️ Downloading model weights...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    model = build_model()
    model.load_weights(MODEL_PATH)

    st.write("✅ Model Loaded Successfully")
    return model

model = load_model()

# =========================
# LABELS
# =========================
class_names = ["Abnormal", "COVID", "MI", "MI History", "Normal"]

# =========================
# IMAGE PREPROCESSING
# =========================
def preprocess_image(img):
    img = img.convert("RGB")

    # 🔥 Crop ECG area (adjust if needed)
    width, height = img.size
    img = img.crop((int(0.05*width), int(0.05*height),
                    int(0.95*width), int(0.95*height)))

    img = img.resize((128,128))

    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# =========================
# UI
# =========================
st.title("ECG Image Classification")

file = st.file_uploader("Upload ECG Image", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file)
    st.image(img, caption="Uploaded ECG", use_container_width=True)

    processed = preprocess_image(img)

    pred = model.predict(processed)
    cls = np.argmax(pred)
    conf = np.max(pred)

    # 🔹 Debug (optional)
    st.write("Prediction probabilities:", pred)

    st.success(f"Prediction: {class_names[cls]}")
    st.write(f"Confidence: {conf:.2f}")
