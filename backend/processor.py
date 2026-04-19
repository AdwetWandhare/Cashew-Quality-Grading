import logging

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from model import IMAGE_SIZE, ModelService

logger = logging.getLogger(__name__)
UNIFORM_THRESHOLD = 0.02


def decode_image(image_bytes: bytes) -> np.ndarray:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode image. Upload a valid JPG, PNG, BMP, or WebP file.")
    return image


def preprocess_image(image: np.ndarray) -> np.ndarray:
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_image, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    float_image = resized.astype("float32")
    processed = preprocess_input(float_image)
    return np.expand_dims(processed, axis=0)


def predict_grade(image_bytes: bytes, model_service: ModelService) -> dict:
    if not model_service.weights_loaded:
        raise RuntimeError("No compatible trained model is loaded. Run train.py for the current dataset first.")

    image = decode_image(image_bytes)
    processed_image = preprocess_image(image)
    probabilities = model_service.model.predict(processed_image, verbose=0)[0]

    if len(probabilities) != len(model_service.class_names):
        raise RuntimeError(
            "Model output does not match labels. "
            f"Model outputs {len(probabilities)} classes, but labels contain {len(model_service.class_names)} classes."
        )

    probability_spread = float(probabilities.max() - probabilities.min())
    if probability_spread < UNIFORM_THRESHOLD:
        logger.warning("Near-uniform prediction detected with spread %.4f.", probability_spread)

    predicted_index = int(np.argmax(probabilities))
    predicted_grade = model_service.class_names[predicted_index]
    confidence = float(probabilities[predicted_index])
    top_indices = np.argsort(probabilities)[::-1][: min(3, len(probabilities))]

    return {
        "predicted_grade": predicted_grade,
        "confidence": round(confidence, 4),
        "probability_spread": round(probability_spread, 4),
        "top_predictions": [
            {
                "grade": model_service.class_names[index],
                "confidence": round(float(probabilities[index]), 4),
            }
            for index in top_indices
        ],
        "class_probabilities": {
            grade: round(float(score), 4)
            for grade, score in zip(model_service.class_names, probabilities)
        },
    }
