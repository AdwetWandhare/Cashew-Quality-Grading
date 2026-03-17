from collections import Counter
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from model import IMAGE_SIZE, LABELS_PATH, MODEL_PATH, build_cnn_model, save_class_names

DATASET_DIR = Path("Dataset")
TRAIN_DIR = DATASET_DIR / "train"
VALID_DIR = DATASET_DIR / "valid"
BEST_MODEL_PATH = Path("cashew_best_model.keras")
GRADING_CLASSES = [f"s{i}" for i in range(22)]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BATCH_SIZE = 16
EPOCHS = 20
AUTOTUNE = tf.data.AUTOTUNE


def collect_samples(split_dir: Path, class_names: list[str]) -> tuple[list[str], list[int]]:
    image_paths: list[str] = []
    labels: list[int] = []

    for label_index, class_name in enumerate(class_names):
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue

        for image_path in class_dir.iterdir():
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(str(image_path))
                labels.append(label_index)

    return image_paths, labels


def decode_and_resize(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def build_dataset(image_paths: list[str], labels: list[int], training: bool) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(decode_and_resize, num_parallel_calls=AUTOTUNE)

    if training:
        dataset = dataset.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)

    return dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)


def validate_dataset(train_paths: list[str], valid_paths: list[str], class_names: list[str]) -> None:
    if not train_paths:
        raise RuntimeError("No training images found for classes s0 to s21 in Dataset/train.")
    if not valid_paths:
        raise RuntimeError("No validation images found for classes s0 to s21 in Dataset/valid.")

    train_counter = Counter(Path(path).parent.name for path in train_paths)
    valid_counter = Counter(Path(path).parent.name for path in valid_paths)

    missing_train = [class_name for class_name in class_names if train_counter[class_name] == 0]
    missing_valid = [class_name for class_name in class_names if valid_counter[class_name] == 0]

    if missing_train:
        raise RuntimeError(f"Missing training images for classes: {', '.join(missing_train)}")
    if missing_valid:
        raise RuntimeError(f"Missing validation images for classes: {', '.join(missing_valid)}")


def main() -> None:
    if not TRAIN_DIR.exists():
        raise RuntimeError(f"Training directory not found: {TRAIN_DIR}")
    if not VALID_DIR.exists():
        raise RuntimeError(f"Validation directory not found: {VALID_DIR}")

    train_paths, train_labels = collect_samples(TRAIN_DIR, GRADING_CLASSES)
    valid_paths, valid_labels = collect_samples(VALID_DIR, GRADING_CLASSES)
    validate_dataset(train_paths, valid_paths, GRADING_CLASSES)

    train_dataset = build_dataset(train_paths, train_labels, training=True)
    valid_dataset = build_dataset(valid_paths, valid_labels, training=False)

    model = build_cnn_model(input_shape=(*IMAGE_SIZE, 3), num_classes=len(GRADING_CLASSES))
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ModelCheckpoint(filepath=str(BEST_MODEL_PATH), monitor="val_accuracy", save_best_only=True),
    ]

    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    model.save(MODEL_PATH)
    save_class_names(GRADING_CLASSES, LABELS_PATH)

    best_val_accuracy = max(history.history.get("val_accuracy", [0.0]))
    print(f"Training complete. Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Model saved to: {MODEL_PATH.resolve()}")
    print(f"Best checkpoint saved to: {BEST_MODEL_PATH.resolve()}")
    print(f"Labels saved to: {LABELS_PATH.resolve()}")


if __name__ == "__main__":
    main()
