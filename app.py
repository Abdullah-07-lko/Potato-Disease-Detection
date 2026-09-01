

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Potato Leaf AI Scanner",
    page_icon="🥔",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 10% 20%, rgba(76,175,80,0.12), transparent 25%),
        radial-gradient(circle at 90% 80%, rgba(0,200,150,0.10), transparent 25%),
        linear-gradient(135deg, #07110d, #0b1712, #06100c);
    color: #f5f7f5;
    min-height: 100vh;
}

.stApp::before {
    content: "";
    position: fixed;
    width: 350px;
    height: 350px;
    border-radius: 50%;
    background: rgba(76,175,80,0.08);
    filter: blur(80px);
    top: 10%;
    left: -100px;
    animation: floatOne 10s ease-in-out infinite;
    pointer-events: none;
}

.stApp::after {
    content: "";
    position: fixed;
    width: 300px;
    height: 300px;
    border-radius: 50%;
    background: rgba(0,230,118,0.06);
    filter: blur(80px);
    bottom: 5%;
    right: -80px;
    animation: floatTwo 12s ease-in-out infinite;
    pointer-events: none;
}

@keyframes floatOne {
    0%,100% { transform: translateY(0); }
    50% { transform: translateY(60px); }
}

@keyframes floatTwo {
    0%,100% { transform: translateY(0); }
    50% { transform: translateY(-70px); }
}

.hero {
    text-align: center;
    padding: 25px 10px 15px;
}

.logo {
    font-size: 4rem;
    animation: bounce 3s ease-in-out infinite;
}

@keyframes bounce {
    0%,100% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
}

.main-title {
    font-size: 2.7rem;
    font-weight: 800;
    background: linear-gradient(90deg,#8cff9b,#48e06f,#b4ffbf);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}

.subtitle {
    color: #a8b8ad;
    font-size: 1.05rem;
    margin-bottom: 20px;
}

.ai-badge {
    display: inline-block;
    padding: 7px 15px;
    border-radius: 30px;
    background: rgba(76,175,80,0.12);
    border: 1px solid rgba(100,220,120,0.25);
    color: #8cff9b;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 12px;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(76,175,80,0.3); }
    70% { box-shadow: 0 0 0 10px rgba(76,175,80,0); }
    100% { box-shadow: 0 0 0 0 rgba(76,175,80,0); }
}

.upload-card {
    background: rgba(20,35,27,0.75);
    border: 1px solid rgba(130,200,140,0.18);
    border-radius: 20px;
    padding: 20px;
    margin-top: 15px;
    box-shadow: 0 10px 35px rgba(0,0,0,0.25);
    backdrop-filter: blur(12px);
}

[data-testid="stFileUploader"] {
    background: rgba(15,27,21,0.8);
    border-radius: 16px;
    padding: 10px;
    border: 1px dashed rgba(120,220,130,0.35);
}

[data-testid="stImage"] img {
    border-radius: 18px;
    border: 1px solid rgba(120,220,130,0.2);
    box-shadow: 0 12px 35px rgba(0,0,0,0.35);
    transition: transform 0.3s ease;
}

[data-testid="stImage"] img:hover {
    transform: scale(1.015);
}

.result-card {
    background: linear-gradient(135deg,rgba(28,55,37,0.95),rgba(14,30,21,0.95));
    border-radius: 20px;
    padding: 25px;
    margin-top: 25px;
    border: 1px solid rgba(130,220,140,0.2);
    box-shadow: 0 15px 40px rgba(0,0,0,0.35);
    animation: slideUp 0.6s ease;
}

@keyframes slideUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.diagnosis-title {
    color: #91ff9d;
    font-size: 0.9rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.diagnosis {
    color: white;
    font-size: 2rem;
    font-weight: 800;
    margin-top: 5px;
}

.confidence-text {
    color: #aebdb3;
    font-size: 0.95rem;
    margin-top: 10px;
}

.progress-container {
    width: 100%;
    height: 12px;
    background: #17251c;
    border-radius: 20px;
    overflow: hidden;
    margin-top: 10px;
}

.progress-bar {
    height: 100%;
    background: linear-gradient(90deg,#39c95a,#8cff9b);
    border-radius: 20px;
    animation: progressAnimation 1.5s ease-out;
}

@keyframes progressAnimation {
    from { width: 0%; }
}

.info-card {
    background: rgba(20,35,27,0.65);
    border: 1px solid rgba(130,200,140,0.15);
    border-radius: 16px;
    padding: 18px;
    margin-top: 20px;
    transition: 0.3s ease;
}

.info-card:hover {
    transform: translateY(-4px);
    border-color: rgba(130,220,140,0.35);
}

.info-title {
    color: #8cff9b;
    font-weight: 700;
    font-size: 1rem;
}

.info-text {
    color: #aab7ae;
    font-size: 0.9rem;
    line-height: 1.6;
}

.footer {
    text-align: center;
    color: #718077;
    font-size: 0.8rem;
    margin-top: 35px;
    padding-bottom: 20px;
}

label {
    color: #dce8df !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/1.keras")

model = load_model()

class_names = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

st.markdown("""
<div class="hero">
    <div class="logo">🥔</div>
    <div class="ai-badge">✨ AI POWERED PLANT HEALTH DETECTION</div>
    <div class="main-title">Potato Leaf AI Scanner</div>
    <div class="subtitle">Detect potato leaf diseases using Deep Learning</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="upload-card">
    <h3 style="color:white; margin-bottom:5px;">📸 Upload Your Leaf</h3>
    <p style="color:#9eada3;">
        Upload a clear image of a potato leaf and let the AI analyze it.
    </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown(
        "<p style='color:#9eada3; margin-top:20px;'>🔍 Uploaded Image</p>",
        unsafe_allow_html=True
    )

    st.image(
        image,
        caption="Potato Leaf",
        use_container_width=True
    )

    img = image.resize((256, 256))
    img_array = np.expand_dims(np.array(img), axis=0)

    with st.spinner("🧠 AI is analyzing the leaf..."):
        prediction = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction[0]) * 100)

    label = (
        predicted_class
        .replace("Potato___", "")
        .replace("_", " ")
        .title()
    )

    if "healthy" in predicted_class.lower():
        icon = "🌿"
        status = "Healthy Leaf"
        message = """
        The leaf appears healthy according to the model.
        Continue maintaining proper irrigation, nutrition,
        and regular monitoring.
        """

    elif "early" in predicted_class.lower():
        icon = "🍂"
        status = "Early Blight Detected"
        message = """
        Early blight symptoms may appear as dark spots
        and concentric rings on the leaf. Consider
        inspecting nearby plants and improving crop care.
        """

    else:
        icon = "⚠️"
        status = "Late Blight Detected"
        message = """
        Late blight can spread rapidly under favorable
        conditions. Inspect the crop carefully and
        consider appropriate plant protection measures.
        """

    st.markdown(f"""
    <div class="result-card">

        <div class="diagnosis-title">
            🤖 AI Diagnosis
        </div>

        <div class="diagnosis">
            {icon} {label}
        </div>

        <div class="confidence-text">
            Model Confidence: <b>{confidence:.2f}%</b>
        </div>

        <div class="progress-container">
            <div class="progress-bar" style="width:{confidence}%;"></div>
        </div>

        <div style="
            margin-top:15px;
            color:#9eada3;
            font-size:0.9rem;
        ">
            Status:
            <b style="color:#8cff9b;">
                {status}
            </b>
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-card">

        <div class="info-title">
            💡 About This Result
        </div>

        <div class="info-text">
            {message}
        </div>

    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="info-card">

    <div class="info-title">
        🧠 How It Works
    </div>

    <div class="info-text">

        <b>1️⃣ Upload</b> — Provide a potato leaf image.<br><br>

        <b>2️⃣ Analyze</b> — The trained CNN model processes
        the image.<br><br>

        <b>3️⃣ Predict</b> — The model identifies the most
        likely leaf condition.<br><br>

        <b>4️⃣ Confidence</b> — The model displays its
        prediction confidence.

    </div>

</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    🥔 Potato Leaf AI Scanner
    <br>
    Powered by TensorFlow & Convolutional Neural Networks
</div>
""", unsafe_allow_html=True)



