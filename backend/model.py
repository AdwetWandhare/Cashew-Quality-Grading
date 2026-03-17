import json
from dataclasses import dataclass
from pathlib import Path

from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential, load_model

DEFAULT_CLASS_NAMES = [f"s{i}" for i in range(22)]
IMAGE_SIZE = (224, 224)
MODEL_PATH = Path("cashew_model.h5")
LABELS_PATH = Path("cashew_labels.json")


@dataclass
class ModelService:
    model: Sequential
    class_names: list[str]
    weights_loaded: bool


def build_cnn_model(input_shape: tuple[int, int, int] = (224, 224, 3), num_classes: int = 22) -> Sequential:
    model = Sequential(
        [
            Input(shape=input_shape),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.4),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def save_class_names(class_names: list[str], labels_path: Path = LABELS_PATH) -> None:
    labels_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")


def load_class_names(labels_path: Path = LABELS_PATH) -> list[str]:
    if labels_path.exists():
        return json.loads(labels_path.read_text(encoding="utf-8"))
    return DEFAULT_CLASS_NAMES


def load_cashew_model(model_path: Path, labels_path: Path = LABELS_PATH) -> ModelService:
    class_names = load_class_names(labels_path)

    if model_path.exists():
        model = load_model(model_path)
        return ModelService(model=model, class_names=class_names, weights_loaded=True)

    model = build_cnn_model(input_shape=(*IMAGE_SIZE, 3), num_classes=len(class_names))
    return ModelService(model=model, class_names=class_names, weights_loaded=False)
