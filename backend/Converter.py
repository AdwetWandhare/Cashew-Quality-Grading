import argparse
import json
import shutil
from pathlib import Path


def convert_coco_split(split_dir: Path, output_dir: Path) -> None:
    annotation_path = split_dir / "_annotations.coco.json"
    if not annotation_path.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {annotation_path}")

    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {category["id"]: category["name"] for category in data.get("categories", [])}
    images = {image["id"]: image["file_name"] for image in data.get("images", [])}

    copied = 0
    skipped = 0
    for annotation in data.get("annotations", []):
        class_name = categories.get(annotation.get("category_id"))
        file_name = images.get(annotation.get("image_id"))

        if not class_name or not file_name:
            skipped += 1
            continue

        source_path = split_dir / file_name
        if not source_path.exists():
            skipped += 1
            continue

        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, class_dir / file_name)
        copied += 1

    print(f"Copied {copied} images into {output_dir}")
    if skipped:
        print(f"Skipped {skipped} annotations with missing category or image data.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a Roboflow COCO split into class folders.")
    parser.add_argument("--split", default="valid", help="Dataset split folder to convert, for example train, valid, or test.")
    parser.add_argument(
        "--dataset-dir",
        default="Dataset",
        help="Dataset root directory. Defaults to Dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output folder. Defaults to Dataset/<split>_sorted.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    split_dir = dataset_dir / args.split
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir / f"{args.split}_sorted"

    convert_coco_split(split_dir=split_dir, output_dir=output_dir)


if __name__ == "__main__":
    main()
