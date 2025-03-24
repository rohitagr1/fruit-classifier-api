import os
import gdown
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
MODEL_PATH = "fruit_classifier.h5"
MODEL_ID = "1pmBmVyZWMuuOml-RRb0fqhlu22cnWpLt"  # Google Drive ID
LABELS = ['freshapples', 'freshbanana', 'freshoranges',
                'rottenapples', 'rottenbanana', 'rottenoranges']

# Download model if not present
if not os.path.exists(MODEL_PATH):
    gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", MODEL_PATH, quiet=False)


# Load model
def load_trained_model():
    global model
    model = load_model(MODEL_PATH)


load_trained_model()


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    file_path = "uploaded_image.jpg"
    file.save(file_path)

    img = preprocess_image(file_path)
    preds = model.predict(img)
    predicted_label = LABELS[np.argmax(preds)]  # Get class label

    return jsonify({'prediction': predicted_label})


if __name__ == '__main__':
    app.run(debug=True)
