
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Model loading
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/1.keras")

model = load_model()

class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

st.title("🥔 Potato Disease Detector")
st.write("Patte ki photo upload karo, model bimari detect karega")

uploaded_file = st.file_uploader("Image upload karo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess 
    img = image.resize((256, 256))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = round(100 * np.max(prediction[0]), 2)

    st.subheader(f"Result: {predicted_class}")
    st.write(f"Confidence: {confidence}%")



