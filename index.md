# **Handwritten Digit Recognition 📝🔢**
**Classifying handwritten digits using a CNN model trained on the MNIST dataset.**  

---

## **Overview**
This project builds a **Convolutional Neural Network (CNN)** to classify handwritten digits (0-9) using the **MNIST dataset**. The model is trained using **TensorFlow/Keras** and achieves high accuracy in recognizing digits from images.  

Additionally, an **API** is provided to perform real-time digit recognition from user-uploaded images.

---

## **📁 Repository Structure**
```
Handwritten-Digit-Recognition/
│── dataset/               # (Optional) Additional dataset files
│── notebooks/             # Jupyter notebooks for EDA & training
│── src/                   # Scripts for model training, evaluation, inference
│── models/                # Saved models and checkpoints
│── results/               # Logs, evaluation reports, and visualizations
│── requirements.txt       # Project dependencies
│── README.md              # Project overview and instructions
│── train.py               # Main script for training the model
│── inference.py           # Script for making predictions
│── app.py                 # Flask/FastAPI app for serving predictions
│── config.yaml            # Model configurations
│── LICENSE                # License file
│── .gitignore             # Ignore unnecessary files
```

---

## **🛠 Installation**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/nagakirankasi/Handwritten-Digit-Recognition.git
cd Handwritten-Digit-Recognition
```

### **2️⃣ Install Dependencies**
Ensure Python 3.8+ is installed, then run:
```bash
pip install -r requirements.txt
```

#### **📌 Dependencies**
```
tensorflow
keras
scikit-learn
matplotlib
numpy
pandas
jupyter
flask  # (Optional, if serving via API)
fastapi  # (Optional, if serving via FastAPI)
uvicorn  # (Optional, if using FastAPI)
```

---

## **📊 Dataset**
We use the **MNIST dataset**, which contains **60,000 training images** and **10,000 test images** of handwritten digits (0-9).  

📌 **Load the dataset in Python:**
```python
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

📌 **Visualizing Sample Digits**
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for i, ax in enumerate(axes):
    ax.imshow(X_train[i], cmap='gray')
    ax.axis('off')
plt.show()
```

---

## **🔧 Preprocessing**
1. **Normalization** – Scale pixel values to [0,1].
2. **Reshaping** – Convert images to `(28,28,1)` for CNN.
3. **One-hot Encoding** – Convert labels into categorical values.

```python
from tensorflow.keras.utils import to_categorical

# Normalize
X_train, X_test = X_train / 255.0, X_test / 255.0

# Reshape for CNN
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# One-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
```

---

## **🏗 Model Architecture**
This project uses a **Convolutional Neural Network (CNN)** with the following architecture:
- **Conv2D** (32 filters, 3×3, ReLU)
- **MaxPooling2D** (2×2)
- **Conv2D** (64 filters, 3×3, ReLU)
- **MaxPooling2D** (2×2)
- **Flatten**
- **Dense** (128 neurons, ReLU)
- **Dropout** (0.5)
- **Dense** (10 neurons, Softmax)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

---

## **🚀 Model Training**
Train the model using:
```bash
python train.py
```
or manually:
```python
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
```
📌 **Save the model:**
```python
model.save("models/mnist_model.h5")
```

---

## **📈 Model Evaluation**
After training, evaluate the model:
```python
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
```
📌 **Confusion Matrix**
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred = model.predict(X_test).argmax(axis=1)
y_true = y_test.argmax(axis=1)

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
```

---

## **🖼️ Inference on New Images**
Run **inference.py** to predict new images:
```bash
python inference.py --image path/to/image.png
```
📌 **inference.py**
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import argparse

# Load the trained model
model = load_model("models/mnist_model.h5")

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
```

---

## **🖥️ Deploying as an API**
### **Streamlit Implementation**
Run the Streamlit app:
```bash
streamlit run app.py
```
📌 **API Endpoint**
```python
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

```

---

## **📜 License**
This project is licensed under the **MIT License**.

---

## **💡 Future Improvements**
✔ Improve accuracy with deeper architectures  
✔ Train on additional handwritten datasets  
✔ Deploy with a UI (Streamlit, Flask)  

---
