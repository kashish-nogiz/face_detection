from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import io

app = FastAPI(title="Mask Detection API")

# Load model once at startup
MODEL_PATH = "mask_nonmask.h5"
model = load_model(MODEL_PATH)
class_names = ["with_mask", "without_mask"]

def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/")
def home():
    return {"message": "Mask Detection API is running!"}

@app.post("/predict/")
async def predict_mask(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img_array = preprocess_image(contents)
        pred = model.predict(img_array, verbose=0)[0][0]
        label = 1 if pred >= 0.5 else 0
        confidence = pred if label == 1 else 1 - pred
        result = {
            "class": class_names[label],
            "confidence": float(confidence)
        }
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
