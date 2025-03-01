from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import uvicorn

app = FastAPI()
model = load_model("models/mnist_model.h5")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28)).reshape(1, 28, 28, 1) / 255.0
    prediction = model.predict(img).argmax()
    return {"digit": int(prediction)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
