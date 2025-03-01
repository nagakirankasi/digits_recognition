import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("models/mnist_model.h5")

def predict_digit(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = img.reshape(1, 28, 28, 1) / 255.0
    prediction = model.predict(img).argmax()
    return prediction

print(predict_digit("sample_digit.png"))
