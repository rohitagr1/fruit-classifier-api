# Fruit Classifier API

This API predicts whether a fruit is rotten or fresh. The model is built using TensorFlow and FastAPI.

## Usage

To use this API, send a `POST` request to the `/predict/` endpoint with a base64 encoded image of the fruit.

### Example Request

```json
{
    "image": "base64encodedimagehere"
}
