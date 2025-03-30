from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import io
from keras.saving import register_keras_serializable

@register_keras_serializable()
def gray_to_rgb(x):
    return x  # Replace this with the actual function logic

# Load the trained model
model = load_model('emotion_model.h5')

# Define emotion labels (order must match training labels)
EMOTIONS = ['happy', 'sad', 'angry', 'surprised', 'neutral']

# Initialize FastAPI app
app = FastAPI()

@app.post("/predict_emotion/")
async def predict_emotion(file: UploadFile = File(...)):
    # Read image file
    image = await file.read()
    image = Image.open(io.BytesIO(image)).convert('L')  # Convert to grayscale
    image = image.resize((48, 48))  # Resize to match model input

    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
    image_array = np.expand_dims(image_array, axis=0)   # Add batch dimension

    # Predict emotion
    preds = model.predict(image_array)[0]
    most_prob_index = np.argmax(preds)
    most_prob_emotion = EMOTIONS[most_prob_index]
    prob_value = preds[most_prob_index]

    # Return response as JSON
    return {
        "emotion": most_prob_emotion,
        "confidence": float(prob_value)  # Convert NumPy float to regular float
    }

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
