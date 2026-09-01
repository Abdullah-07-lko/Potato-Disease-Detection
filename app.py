

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Potato Leaf Health Scanner", page_icon="🥔", layout="centered")

st.markdown("""
<style>
@keyframes drift {
    0% { background-position: 0% 0%; }
    100% { background-position: 100% 100%; }
}
.stApp {
    background: linear-gradient(120deg, #f4f9f1, #e8f3e3, #f4f9f1);
    background-size: 200% 200%;
    animation: drift 18s ease infinite;
}
.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #2d4a2b;
    text-align: center;
    margin-bottom: 0.2rem;
}
.subtitle {
    text-align: center;
    color: #5a6b57;
    font-size: 1.05rem;
    margin-bottom: 2rem;
}
.result-card {
    background: #ffffffcc;
    border-radius: 14px;
    padding: 1.5rem;
    margin-top: 1.5rem;
    border-left: 6px solid #4a7c3f;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/1.keras")

model = load_model()

class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

st.markdown('<div class="main-title">🥔 Potato Leaf Health Scanner</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a photo of a potato leaf to check for early signs of disease</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your uploaded leaf", use_container_width=True)

    img = image.resize((256, 256))
    img_array = np.expand_dims(np.array(img), axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = round(100 * np.max(prediction[0]), 2)

    label = predicted_class.replace("Potato___", "").replace("_", " ")

    st.markdown(f"""
    <div class="result-card">
        <h4 style="margin:0; color:#2d4a2b;">Diagnosis: {label}</h4>
        <p style="margin:0.3rem 0 0 0; color:#5a6b57;">Confidence: {confidence}%</p>
    </div>
    """, unsafe_allow_html=True)



