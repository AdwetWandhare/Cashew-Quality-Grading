import json
import os
import re
from collections import Counter
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from model import (
    BEST_MODEL_PATH,
    IMAGE_SIZE,
    LABELS_PATH,
    MODEL_PATH,
    build_transfer_model,
    save_class_names,
    unfreeze_and_finetune,
)

DATASET_DIR = Path("Dataset")
TRAIN_DIR = DATASET_DIR / "train"
VALID_DIR = DATASET_DIR / "valid"
TEST_DIR = DATASET_DIR / "test"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BATCH_SIZE = 16
HEAD_EPOCHS = 15
FINE_TUNE_EPOCHS = 20
AUTOTUNE = tf.data.AUTOTUNE

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

augment_pipeline = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.12),
        tf.keras.layers.RandomZoom(0.12),
        tf.keras.layers.RandomContrast(0.15),
    ],
    name="augmentation",
)


def natural_sort_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def list_image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []

    return sorted(
        [
            child
            for child in directory.iterdir()
            if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: natural_sort_key(path.name),
    )


def discover_class_names(train_dir: Path = TRAIN_DIR) -> list[str]:
    folder_classes = [
        child.name
        for child in train_dir.iterdir()
        if child.is_dir() and list_image_files(child)
    ] if train_dir.exists() else []

    if folder_classes:
        return sorted(folder_classes, key=natural_sort_key)

    annotation_path = train_dir / "_annotations.coco.json"
    if annotation_path.exists():
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        category_names = {category["id"]: category["name"] for category in data.get("categories", [])}
        used_category_ids = {annotation["category_id"] for annotation in data.get("annotations", [])}
        coco_classes = [
            category_names[category_id]
            for category_id in used_category_ids
            if category_id in category_names
        ]
        if coco_classes:
            return sorted(set(coco_classes), key=natural_sort_key)

    raise RuntimeError(
        "Could not discover classes. Use Dataset/train/<class_name>/*.jpg "
        "or Dataset/train/_annotations.coco.json."
    )


def collect_folder_samples(split_dir: Path, class_names: list[str]) -> tuple[list[str], list[int]]:
    paths: list[str] = []
    labels: list[int] = []

    for label_index, class_name in enumerate(class_names):
        class_dir = split_dir / class_name
        for image_path in list_image_files(class_dir):
            paths.append(str(image_path))
            labels.append(label_index)

    return paths, labels


def collect_coco_samples(split_dir: Path, class_names: list[str]) -> tuple[list[str], list[int]]:
    annotation_path = split_dir / "_annotations.coco.json"
    if not annotation_path.exists():
        return [], []

    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    category_names = {category["id"]: category["name"] for category in data.get("categories", [])}
    image_names = {image["id"]: image["file_name"] for image in data.get("images", [])}
    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
    image_to_label: dict[str, int] = {}
    skipped_unknown = 0
    skipped_missing = 0
    skipped_ambiguous = 0

    for annotation in data.get("annotations", []):
        category_name = category_names.get(annotation.get("category_id"))
        if category_name not in class_to_index:
            skipped_unknown += 1
            continue

        file_name = image_names.get(annotation.get("image_id"))
        if not file_name:
            skipped_missing += 1
            continue

        image_path = split_dir / file_name
        if not image_path.exists():
            skipped_missing += 1
            continue

        label_index = class_to_index[category_name]
        previous_label = image_to_label.get(file_name)
        if previous_label is not None and previous_label != label_index:
            skipped_ambiguous += 1
            continue

        image_to_label[file_name] = label_index

    if skipped_unknown:
        print(f"[data] {split_dir.name}: skipped {skipped_unknown} annotations outside selected classes.")
    if skipped_missing:
        print(f"[data] {split_dir.name}: skipped {skipped_missing} annotations with missing image data.")
    if skipped_ambiguous:
        print(f"[data] {split_dir.name}: skipped {skipped_ambiguous} ambiguous multi-label annotations.")

    paths = [str(split_dir / file_name) for file_name in sorted(image_to_label, key=natural_sort_key)]
    labels = [image_to_label[file_name] for file_name in sorted(image_to_label, key=natural_sort_key)]
    return paths, labels


def collect_samples(split_dir: Path, class_names: list[str]) -> tuple[list[str], list[int], str]:
    folder_paths, folder_labels = collect_folder_samples(split_dir, class_names)
    if folder_paths:
        return folder_paths, folder_labels, "class folders"

    coco_paths, coco_labels = collect_coco_samples(split_dir, class_names)
    if coco_paths:
        return coco_paths, coco_labels, "COCO annotations"

    return [], [], "none"


def decode_and_resize(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label


def apply_augmentation(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    return augment_pipeline(image, training=True), label


def apply_preprocessing(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    return preprocess_input(image), label


def make_dataset(paths: list[str], labels: list[int], training: bool) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)

    dataset = dataset.map(decode_and_resize, num_parallel_calls=AUTOTUNE)

    if training:
        dataset = dataset.map(apply_augmentation, num_parallel_calls=AUTOTUNE)

    return dataset.map(apply_preprocessing, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)


def compute_class_weights(labels: list[int], num_classes: int) -> dict[int, float]:
    counter = Counter(labels)
    total = len(labels)
    return {
        class_index: total / (num_classes * max(counter.get(class_index, 1), 1))
        for class_index in range(num_classes)
    }


def print_distribution(title: str, labels: list[int], class_names: list[str], source: str) -> None:
    counter = Counter(labels)
    print(f"\n[{title}] source: {source}")
    print("-" * 52)
    for index, class_name in enumerate(class_names):
        count = counter.get(index, 0)
        bar = "#" * max(1, count // 10) if count else ""
        print(f"{class_name:>8}: {count:>4} {bar}")
    print(f"{'total':>8}: {len(labels):>4}")
    print("-" * 52)


def require_samples(split_name: str, paths: list[str]) -> None:
    if not paths:
        raise RuntimeError(
            f"No {split_name} images found. Check Dataset/{split_name} folder structure "
            "or _annotations.coco.json."
        )


def best_accuracy(history: tf.keras.callbacks.History) -> float:
    return max(history.history.get("val_accuracy", [0.0]))


def main() -> None:
    if not TRAIN_DIR.exists():
        raise RuntimeError(f"Training directory not found: {TRAIN_DIR}")
    if not VALID_DIR.exists():
        raise RuntimeError(f"Validation directory not found: {VALID_DIR}")

    class_names = discover_class_names(TRAIN_DIR)
    save_class_names(class_names, LABELS_PATH)
    print(f"[data] Classes: {', '.join(class_names)}")

    train_paths, train_labels, train_source = collect_samples(TRAIN_DIR, class_names)
    valid_paths, valid_labels, valid_source = collect_samples(VALID_DIR, class_names)
    test_paths, test_labels, test_source = collect_samples(TEST_DIR, class_names) if TEST_DIR.exists() else ([], [], "none")

    require_samples("train", train_paths)
    require_samples("valid", valid_paths)

    print_distribution("train", train_labels, class_names, train_source)
    print_distribution("valid", valid_labels, class_names, valid_source)
    if test_paths:
        print_distribution("test", test_labels, class_names, test_source)

    train_dataset = make_dataset(train_paths, train_labels, training=True)
    valid_dataset = make_dataset(valid_paths, valid_labels, training=False)
    test_dataset = make_dataset(test_paths, test_labels, training=False) if test_paths else None
    class_weights = compute_class_weights(train_labels, len(class_names))

    print("\n[train] Phase 1: training classification head")
    model = build_transfer_model(num_classes=len(class_names), pretrained=True)
    phase_1_callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5, mode="max", restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(BEST_MODEL_PATH), monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7, verbose=1),
    ]
    phase_1_history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=HEAD_EPOCHS,
        class_weight=class_weights,
        callbacks=phase_1_callbacks,
    )
    phase_1_best = best_accuracy(phase_1_history)

    print("\n[train] Phase 2: fine-tuning MobileNetV2 top layers")
    model = unfreeze_and_finetune(model, num_unfreeze=40)
    phase_2_callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=6, mode="max", restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(BEST_MODEL_PATH), monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-8, verbose=1),
    ]
    phase_2_history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=phase_2_callbacks,
    )
    phase_2_best = best_accuracy(phase_2_history)
    overall_best = max(phase_1_best, phase_2_best)

    test_accuracy = None
    if test_dataset is not None:
        _, test_accuracy = model.evaluate(test_dataset, verbose=0)

    model.save(MODEL_PATH)
    save_class_names(class_names, LABELS_PATH)

    print("\n" + "=" * 60)
    print(f"Classes          : {', '.join(class_names)}")
    print(f"Phase 1 val acc  : {phase_1_best * 100:.2f}%")
    print(f"Phase 2 val acc  : {phase_2_best * 100:.2f}%")
    print(f"Best val acc     : {overall_best * 100:.2f}%")
    if test_accuracy is not None:
        print(f"Test accuracy    : {test_accuracy * 100:.2f}%")
    print(f"Model saved      : {MODEL_PATH.resolve()}")
    print(f"Best checkpoint  : {BEST_MODEL_PATH.resolve()}")
    print(f"Labels saved     : {LABELS_PATH.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
