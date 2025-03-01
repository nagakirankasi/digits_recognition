import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import argparse

# Load the trained model
model = load_model("models/my_mnist_model.h5")

def preprocess_image(image_path):
    """Load and preprocess an image for model prediction."""
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img = np.array(img) / 255.0  # Normalize pixel values
    img = img.reshape(1, 28, 28, 1)  # Reshape for model input
    return img

def predict_digit(image_path):
    """Predict the digit in the given image."""
    img = preprocess_image(image_path)
    prediction = model.predict(img).argmax()
    return prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handwritten Digit Recognition Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    args = parser.parse_args()
    
    digit = predict_digit(args.image)
    print(f"Predicted Digit: {digit}")