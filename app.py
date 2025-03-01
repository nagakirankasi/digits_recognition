import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("models/mnist_model.h5")

def preprocess_image(image):
    """Preprocess the uploaded image for prediction."""
    img = image.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img = np.array(img) / 255.0  # Normalize pixel values
    img = img.reshape(1, 28, 28, 1)  # Reshape for model input
    return img

def predict_digit(image):
    """Predict the digit from the uploaded image."""
    img = preprocess_image(image)
    prediction = model.predict(img).argmax()
    return prediction

# Streamlit UI
st.title("Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit, and the model will predict the number.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Predicting...")
    
    digit = predict_digit(image)
    st.write(f"Predicted Digit: {digit}")
