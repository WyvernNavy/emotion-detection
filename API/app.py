from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import io
from tensorflow.keras.saving import register_keras_serializable
import os

@register_keras_serializable()
def gray_to_rgb(x):
    return x  # Replace this with the actual function logic


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Load the trained model


# Define the correct path
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the model file
model_path = os.path.join(current_dir, "emotion_model.h5")

# Ensure the file exists before loading
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

# Load the model
model = load_model(model_path)



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
