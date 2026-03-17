import cv2
import numpy as np

from model import IMAGE_SIZE, ModelService


def decode_image(image_bytes: bytes) -> np.ndarray:
    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image. Please upload a valid image file.")
    return image


def preprocess_image(image: np.ndarray) -> np.ndarray:
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    resized = cv2.resize(denoised, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0
    batched = np.expand_dims(normalized, axis=0)
    return batched


def predict_grade(image_bytes: bytes, model_service: ModelService) -> dict:
    if not model_service.weights_loaded:
        raise RuntimeError(
            "Model weights file 'cashew_model.h5' was not found. Run train.py or add the trained model file before predictions."
        )

    image = decode_image(image_bytes)
    processed_image = preprocess_image(image)
    probabilities = model_service.model.predict(processed_image, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_grade = model_service.class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    return {
        "predicted_grade": predicted_grade,
        "confidence": round(confidence, 4),
        "class_probabilities": {
            grade: round(float(score), 4)
            for grade, score in zip(model_service.class_names, probabilities)
        },
    }
