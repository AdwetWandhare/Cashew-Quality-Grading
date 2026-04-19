import json
from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import load_model

DEFAULT_CLASS_NAMES = ["W180", "W210", "W300", "W500"]
IMAGE_SIZE = (224, 224)
MODEL_PATH = Path("cashew_model.h5")
BEST_MODEL_PATH = Path("cashew_best_model.keras")
LABELS_PATH = Path("cashew_labels.json")


@dataclass
class ModelService:
    model: Model
    class_names: list[str]
    weights_loaded: bool
    model_path: str | None = None
    output_classes: int | None = None


def compile_classifier(model: Model, learning_rate: float = 1e-3) -> Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_mobilenet_base(pretrained: bool) -> Model:
    weights = "imagenet" if pretrained else None

    try:
        return MobileNetV2(
            input_shape=(*IMAGE_SIZE, 3),
            include_top=False,
            weights=weights,
        )
    except Exception as exc:
        if not pretrained:
            raise

        print(f"[model] Could not load ImageNet weights: {exc}")
        print("[model] Falling back to MobileNetV2 with random weights.")
        return MobileNetV2(
            input_shape=(*IMAGE_SIZE, 3),
            include_top=False,
            weights=None,
        )


def build_transfer_model(num_classes: int = len(DEFAULT_CLASS_NAMES), pretrained: bool = True) -> Model:
    if num_classes < 2:
        raise ValueError("At least two classes are required for classification.")

    base = build_mobilenet_base(pretrained=pretrained)
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="cashew_w_grade_classifier")
    return compile_classifier(model, learning_rate=1e-3)


def unfreeze_and_finetune(model: Model, num_unfreeze: int = 40) -> Model:
    base_model = next(
        (
            layer
            for layer in model.layers
            if isinstance(layer, Model) and "mobilenetv2" in layer.name.lower()
        ),
        None,
    )

    if base_model is None:
        print("[model] MobileNetV2 base layer not found. Skipping fine-tune unfreeze.")
        return compile_classifier(model, learning_rate=1e-5)

    base_model.trainable = True
    for layer in base_model.layers[:-num_unfreeze]:
        layer.trainable = False

    trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
    print(f"[model] Unfroze {trainable_count} MobileNetV2 layers for fine-tuning.")
    return compile_classifier(model, learning_rate=1e-5)


def build_cnn_model(input_shape: tuple[int, int, int] = (*IMAGE_SIZE, 3), num_classes: int = len(DEFAULT_CLASS_NAMES)) -> Model:
    return build_transfer_model(num_classes=num_classes)


def save_class_names(class_names: list[str], labels_path: Path = LABELS_PATH) -> None:
    labels_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")


def load_class_names(labels_path: Path = LABELS_PATH) -> list[str]:
    if not labels_path.exists():
        return DEFAULT_CLASS_NAMES

    try:
        class_names = json.loads(labels_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return DEFAULT_CLASS_NAMES

    if not isinstance(class_names, list) or not all(isinstance(name, str) for name in class_names):
        return DEFAULT_CLASS_NAMES

    return class_names or DEFAULT_CLASS_NAMES


def get_model_output_count(model: Model) -> int | None:
    output_shape = model.output_shape
    if isinstance(output_shape, list):
        output_shape = output_shape[0]

    try:
        return int(output_shape[-1])
    except (TypeError, ValueError, IndexError):
        return None


def load_compatible_model(model_path: Path, class_names: list[str]) -> Model | None:
    if not model_path.exists():
        return None

    try:
        model = load_model(model_path, compile=False)
    except Exception as exc:
        print(f"[model] Could not load {model_path}: {exc}")
        return None

    output_count = get_model_output_count(model)
    if output_count != len(class_names):
        print(
            "[model] Ignoring incompatible model "
            f"{model_path} because it outputs {output_count} classes "
            f"but labels contain {len(class_names)} classes."
        )
        return None

    compile_classifier(model, learning_rate=1e-5)
    print(f"[model] Loaded compatible model from {model_path.resolve()}")
    return model


def load_cashew_model(
    model_path: Path = MODEL_PATH,
    labels_path: Path = LABELS_PATH,
) -> ModelService:
    class_names = load_class_names(labels_path)

    for candidate_path in (model_path, BEST_MODEL_PATH):
        model = load_compatible_model(candidate_path, class_names)
        if model is not None:
            return ModelService(
                model=model,
                class_names=class_names,
                weights_loaded=True,
                model_path=str(candidate_path),
                output_classes=get_model_output_count(model),
            )

    print("[model] No compatible trained model found. Run train.py for the current dataset.")
    model = build_transfer_model(num_classes=len(class_names), pretrained=False)
    return ModelService(
        model=model,
        class_names=class_names,
        weights_loaded=False,
        model_path=None,
        output_classes=get_model_output_count(model),
    )
