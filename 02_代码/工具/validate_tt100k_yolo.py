from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the deterministic TT100K YOLO dataset before training."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    required = [
        root / "tt100k.yaml",
        root / "class_selection.json",
        root / "dataset_manifest.json",
        root / "manifests" / "dataset_summary.json",
        root / "dataset_statistics.csv",
        root / "manifests" / "split_manifest.json",
        root / "manifests" / "class_to_id.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset files: {missing}")

    selection = json.loads((root / "class_selection.json").read_text(encoding="utf-8"))
    class_to_id = {
        str(key): int(value) for key, value in selection["class_to_id"].items()
    }
    if len(class_to_id) < 10:
        raise RuntimeError(f"Expected at least 10 classes, found {len(class_to_id)}")
    if set(class_to_id.values()) != set(range(len(class_to_id))):
        raise RuntimeError("Class IDs are not contiguous from zero")
    dataset_yaml = yaml.safe_load((root / "tt100k.yaml").read_text(encoding="utf-8"))
    if dataset_yaml.get("names") != selection["classes"]:
        raise RuntimeError("tt100k.yaml names differ from class_selection.json")

    image_stems: dict[str, str] = {}
    box_counts: dict[str, Counter] = {}
    label_hash = hashlib.sha256()

    for split in SPLITS:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        images = sorted(
            path
            for path in image_dir.iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        )
        labels = sorted(
            path for path in label_dir.iterdir() if path.suffix.lower() == ".txt"
        )
        image_map = {path.stem: path for path in images}
        label_map = {path.stem: path for path in labels}
        if set(image_map) != set(label_map):
            missing_labels = sorted(set(image_map) - set(label_map))[:10]
            missing_images = sorted(set(label_map) - set(image_map))[:10]
            raise RuntimeError(
                f"{split}: image/label mismatch; missing labels={missing_labels}, "
                f"missing images={missing_images}"
            )
        if not images:
            raise RuntimeError(f"{split}: no images")

        counts: Counter = Counter()
        for stem in sorted(image_map):
            if stem in image_stems:
                raise RuntimeError(
                    f"Image {stem} appears in both {image_stems[stem]} and {split}"
                )
            image_stems[stem] = split
            label_path = label_map[stem]
            raw = label_path.read_bytes()
            label_hash.update(split.encode("utf-8"))
            label_hash.update(stem.encode("utf-8"))
            label_hash.update(b"\0")
            label_hash.update(raw)
            label_hash.update(b"\0")

            text = raw.decode("utf-8")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5:
                    raise RuntimeError(
                        f"{label_path}:{line_number}: expected 5 fields, found {len(fields)}"
                    )
                class_id = int(fields[0])
                if class_id not in class_to_id.values():
                    raise RuntimeError(
                        f"{label_path}:{line_number}: unknown class id {class_id}"
                    )
                x_center, y_center, width, height = map(float, fields[1:])
                values = (x_center, y_center, width, height)
                if not all(0.0 <= value <= 1.0 for value in values):
                    raise RuntimeError(
                        f"{label_path}:{line_number}: coordinates outside [0, 1]"
                    )
                if width <= 0.0 or height <= 0.0:
                    raise RuntimeError(
                        f"{label_path}:{line_number}: non-positive box size"
                    )
                counts[class_id] += 1
        box_counts[split] = counts

    with (root / "dataset_statistics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(class_to_id):
        raise RuntimeError(
            f"Statistics row mismatch: expected {len(class_to_id)}, found {len(rows)}"
        )
    for row in rows:
        class_id = int(row["class_id"])
        for split in SPLITS:
            declared = int(row[f"{split}_boxes"])
            actual = box_counts[split][class_id]
            if declared != actual:
                raise RuntimeError(
                    f"Class {class_id}, split {split}: declared={declared}, actual={actual}"
                )

    summary = json.loads(
        (root / "manifests" / "dataset_summary.json").read_text(encoding="utf-8")
    )
    selected = summary["selected_boxes"]
    actual_totals = {split: sum(box_counts[split].values()) for split in SPLITS}
    if selected != actual_totals:
        raise RuntimeError(
            f"Selected-box totals mismatch: {selected} != {actual_totals}"
        )

    manifest_path = root / "manifests" / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_ids = {
        split: {str(row["image_id"]) for row in manifest if row["split"] == split}
        for split in SPLITS
    }
    if manifest_ids["train"] & manifest_ids["val"]:
        raise RuntimeError("split_manifest contains train/validation overlap")
    if manifest_ids["train"] & manifest_ids["test"]:
        raise RuntimeError("split_manifest contains train/test overlap")
    if manifest_ids["val"] & manifest_ids["test"]:
        raise RuntimeError("split_manifest contains validation/test overlap")
    if sum(len(ids) for ids in manifest_ids.values()) != len(manifest):
        raise RuntimeError("split_manifest contains duplicate image_id rows")

    report = {
        "dataset_root": str(root),
        "classes": len(class_to_id),
        "images": {
            split: sum(1 for value in image_stems.values() if value == split)
            for split in SPLITS
        },
        "boxes": actual_totals,
        "manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
        "class_to_id_sha256": sha256_bytes(
            (root / "manifests" / "class_to_id.json").read_bytes()
        ),
        "labels_tree_sha256": label_hash.hexdigest(),
        "validation": "passed",
    }
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
