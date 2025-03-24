import os
import gdown
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from io import BytesIO
from PIL import Image

app = FastAPI()
MODEL_PATH = "fruit_classifier.h5"
MODEL_ID = "1pmBmVyZWMuuOml-RRb0fqhlu22cnWpLt"  # Google Drive ID
LABELS = ['freshapples', 'freshbanana', 'freshoranges',
          'rottenapples', 'rottenbanana', 'rottenoranges']

model = None  # Initialize model as None


def get_model():
    """Load model only when needed (lazy loading)."""
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", MODEL_PATH, quiet=False)
        model = load_model(MODEL_PATH)
    return model


def preprocess_image(file):
    """Preprocess image directly from memory (no file saving)."""
    img = Image.open(BytesIO(file)).resize((224, 224))  # Adjust size as per model input
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Handle image upload and return prediction."""
    try:
        # Read file as bytes
        img_data = await file.read()

        # Preprocess image
        img = preprocess_image(img_data)

        # Load model only if needed
        model = get_model()

        # Make prediction
        preds = model.predict(img)
        predicted_label = LABELS[np.argmax(preds)]  # Get predicted class

        return JSONResponse(content={'prediction': predicted_label})
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=400)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
