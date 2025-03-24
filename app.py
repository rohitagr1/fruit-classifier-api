import os
import gdown
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from io import BytesIO
from PIL import Image

app = Flask(__name__)
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
    img = Image.open(BytesIO(file.read())).resize((224, 224))  # Adjust size as per model input
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return prediction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    img = preprocess_image(request.files['file'])  # Process image
    model = get_model()  # Load model only if needed
    preds = model.predict(img)
    predicted_label = LABELS[np.argmax(preds)]  # Get predicted class

    return jsonify({'prediction': predicted_label})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
