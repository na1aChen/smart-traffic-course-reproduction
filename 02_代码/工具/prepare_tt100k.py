from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

# The official TT100K tutorial exposes these 45 benchmark classes through
# anno_func.type45. The script validates their training instances after loading
# the official annotations instead of selecting classes from validation/test.
TYPE45 = [
    "i2",
    "i4",
    "i5",
    "il100",
    "il60",
    "il80",
    "io",
    "ip",
    "p10",
    "p11",
    "p12",
    "p19",
    "p23",
    "p26",
    "p27",
    "p3",
    "p5",
    "p6",
    "pg",
    "ph4",
    "ph4.5",
    "ph5",
    "pl100",
    "pl120",
    "pl20",
    "pl30",
    "pl40",
    "pl5",
    "pl50",
    "pl60",
    "pl70",
    "pl80",
    "pm20",
    "pm30",
    "pm55",
    "pn",
    "pne",
    "po",
    "pr40",
    "w13",
    "w32",
    "w55",
    "w57",
    "w59",
    "wo",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the official TT100K annotations to a deterministic YOLO layout."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--archive-sha256")
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--link-mode",
        choices=["hardlink", "copy", "none"],
        default="hardlink",
        help="Use hard links by default to avoid duplicating roughly 18 GiB of images.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def find_single(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {pattern} under {root}, found {len(matches)}"
        )
    return matches[0]


def read_ids(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stable_rank(seed: int, image_id: str) -> str:
    return hashlib.sha256(f"{seed}:{image_id}".encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_image(data_root: Path, split: str, image_id: str, image_info: dict) -> Path:
    candidates = sorted(
        path
        for path in (data_root / split).glob(f"{image_id}.*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(candidates) == 1:
        return candidates[0]
    annotation_path = data_root / Path(image_info["path"])
    if annotation_path.exists():
        return annotation_path
    raise FileNotFoundError(
        f"Cannot resolve image id={image_id} split={split}: candidates={candidates}, "
        f"annotation_path={annotation_path}"
    )


def reset_output(root: Path, force: bool) -> None:
    if root.exists() and any(root.iterdir()):
        if not force:
            raise RuntimeError(
                f"Output directory is not empty: {root}; pass --force to replace owned outputs"
            )
        allowed_dirs = {"images", "labels", "manifests", "stats"}
        allowed_files = {
            "tt100k.yaml",
            "dataset_statistics.csv",
            "dataset_manifest.json",
            "class_selection.json",
        }
        unexpected = [
            p.name for p in root.iterdir() if p.name not in allowed_dirs | allowed_files
        ]
        if unexpected:
            raise RuntimeError(
                f"Refusing to clear unexpected files in {root}: {unexpected}"
            )
        for name in allowed_dirs:
            path = root / name
            if path.exists():
                shutil.rmtree(path)
        for name in allowed_files:
            path = root / name
            if path.exists():
                path.unlink()
    for name in ["images", "labels", "manifests", "stats"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def link_or_copy(source: Path, target: Path, mode: str) -> None:
    if target.exists():
        target.unlink()
    if mode == "none":
        return
    if mode == "hardlink":
        try:
            os.link(source, target)
            return
        except OSError as exc:
            raise RuntimeError(
                f"Hard link failed for {source} -> {target}. "
                "Use --link-mode copy if source and output are on different volumes."
            ) from exc
    shutil.copy2(source, target)


def convert_split(
    split: str,
    image_ids: list[str],
    images: dict[str, dict],
    data_root: Path,
    output_root: Path,
    class_to_id: dict[str, int],
    link_mode: str,
    image_hashes: dict[tuple[str, str], str],
) -> tuple[list[dict], Counter, dict[str, set[str]]]:
    manifest: list[dict] = []
    split_counts: Counter = Counter()
    class_images: dict[str, set[str]] = defaultdict(set)

    for image_id in image_ids:
        image_info = images.get(image_id)
        if image_info is None:
            raise KeyError(f"Image id {image_id} is missing from annotations")

        source_image = resolve_image(data_root, split, image_id, image_info)

        with Image.open(source_image) as image:
            width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid image shape for {source_image}: {(width, height)}"
            )

        target_image = output_root / "images" / split / source_image.name
        target_label = output_root / "labels" / split / f"{source_image.stem}.txt"
        if target_image.exists() or target_label.exists():
            raise RuntimeError(
                f"Output basename collision in {split}: {source_image.name} ({image_id})"
            )
        link_or_copy(source_image, target_image, link_mode)

        yolo_lines: list[str] = []
        class_counter: Counter = Counter()
        for obj in image_info.get("objects", []):
            category = obj.get("category")
            if category not in class_to_id:
                continue
            bbox = obj.get("bbox", {})
            xmin = max(0.0, min(float(bbox["xmin"]), float(width)))
            xmax = max(0.0, min(float(bbox["xmax"]), float(width)))
            ymin = max(0.0, min(float(bbox["ymin"]), float(height)))
            ymax = max(0.0, min(float(bbox["ymax"]), float(height)))
            box_width = xmax - xmin
            box_height = ymax - ymin
            if box_width <= 0 or box_height <= 0:
                continue
            x_center = (xmin + xmax) / (2.0 * width)
            y_center = (ymin + ymax) / (2.0 * height)
            norm_width = box_width / width
            norm_height = box_height / height
            yolo_lines.append(
                f"{class_to_id[category]} {x_center:.8f} {y_center:.8f} "
                f"{norm_width:.8f} {norm_height:.8f}"
            )
            class_counter[category] += 1
            class_images[category].add(image_id)

        target_label.write_text(
            "\n".join(yolo_lines) + ("\n" if yolo_lines else ""),
            encoding="utf-8",
        )
        split_counts.update(class_counter)
        manifest.append(
            {
                "image_id": image_id,
                "split": split,
                "source_image": str(source_image),
                "output_image": str(target_image),
                "label": str(target_label),
                "image_sha256": image_hashes[(split, image_id)],
                "image_width": width,
                "image_height": height,
                "boxes_selected": sum(class_counter.values()),
            }
        )

    return manifest, split_counts, class_images


def main() -> None:
    args = parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")
    if len(TYPE45) < 10 or len(TYPE45) != len(set(TYPE45)):
        raise AssertionError("TYPE45 must contain at least 10 unique classes")

    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    annotation_path = find_single(data_root, "annotations.json")

    train_ids_path = (
        find_single(data_root, "ids.txt")
        if len(list(data_root.rglob("ids.txt"))) == 1
        else None
    )
    if train_ids_path is None:
        train_ids_path = find_single(data_root / "train", "ids.txt")
    test_ids_path = find_single(data_root / "test", "ids.txt")

    all_data = json.loads(annotation_path.read_text(encoding="utf-8"))
    images: dict[str, dict] = {str(k): v for k, v in all_data["imgs"].items()}
    missing_types = sorted(set(TYPE45) - set(all_data["types"]))
    if missing_types:
        raise RuntimeError(
            f"Annotation types are missing official TYPE45 classes: {missing_types}"
        )
    train_pool = read_ids(train_ids_path)
    test_ids = read_ids(test_ids_path)
    if len(train_pool) != len(set(train_pool)):
        raise RuntimeError("Duplicate ids in train ids.txt")
    if len(test_ids) != len(set(test_ids)):
        raise RuntimeError("Duplicate ids in test ids.txt")
    overlap = set(train_pool) & set(test_ids)
    overlap_exclusions = []
    for image_id in sorted(overlap):
        image_path = resolve_image(data_root, "test", image_id, images[image_id])
        overlap_exclusions.append(
            {
                "image_id": image_id,
                "image_sha256": sha256_file(image_path),
                "reason": "official train/test id overlap; kept in test only",
            }
        )

    test_hashes: dict[str, str] = {}
    test_hash_duplicates: list[tuple[str, str]] = []
    for image_id in test_ids:
        image_path = resolve_image(data_root, "test", image_id, images[image_id])
        digest = sha256_file(image_path)
        if digest in test_hashes:
            test_hash_duplicates.append((digest, image_id))
        else:
            test_hashes[digest] = image_id

    content_exclusions = []
    retained_train_pool = []
    for image_id in train_pool:
        if image_id in overlap:
            continue
        image_path = resolve_image(data_root, "train", image_id, images[image_id])
        digest = sha256_file(image_path)
        if digest in test_hashes:
            content_exclusions.append(
                {
                    "image_id": image_id,
                    "matching_test_image_id": test_hashes[digest],
                    "image_sha256": digest,
                    "reason": "exact image content appears in test; excluded from training pool",
                }
            )
        else:
            retained_train_pool.append(image_id)

    ranked_train = sorted(
        retained_train_pool, key=lambda image_id: stable_rank(args.seed, image_id)
    )
    val_size = max(1, round(len(ranked_train) * args.val_fraction))
    val_ids = sorted(ranked_train[:val_size])
    train_ids = sorted(ranked_train[val_size:])
    if set(train_ids) & set(val_ids):
        raise AssertionError("Train/validation split overlap")

    reset_output(output_root, args.force)
    class_to_id = {name: index for index, name in enumerate(TYPE45)}

    all_manifest: list[dict] = []
    counts: dict[str, Counter] = {}
    image_sets: dict[str, dict[str, set[str]]] = {}
    image_hashes: dict[tuple[str, str], str] = {}
    for split, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        for image_id in ids:
            image_path = resolve_image(data_root, split, image_id, images[image_id])
            image_hashes[(split, image_id)] = sha256_file(image_path)
    for split, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        manifest, split_counts, class_images = convert_split(
            split=split,
            image_ids=ids,
            images=images,
            data_root=data_root,
            output_root=output_root,
            class_to_id=class_to_id,
            link_mode=args.link_mode,
            image_hashes=image_hashes,
        )
        all_manifest.extend(manifest)
        counts[split] = split_counts
        image_sets[split] = class_images

    empty_train_classes = [name for name in TYPE45 if counts["train"][name] == 0]
    if empty_train_classes:
        raise RuntimeError(f"Classes without training instances: {empty_train_classes}")

    data_yaml = {
        "path": str(output_root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": TYPE45,
    }
    (output_root / "tt100k.yaml").write_text(
        yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (output_root / "manifests" / "split_manifest.json").write_text(
        json.dumps(all_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "manifests" / "leakage_exclusions.json").write_text(
        json.dumps(
            {
                "train_test_id_overlap": overlap_exclusions,
                "train_test_content_overlap": content_exclusions,
                "test_internal_sha256_duplicates": [
                    {"sha256": digest, "duplicate_image_id": image_id}
                    for digest, image_id in test_hash_duplicates
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_root / "manifests" / "class_to_id.json").write_text(
        json.dumps(class_to_id, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stats_path = output_root / "stats" / "class_statistics.csv"
    with stats_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "class_id",
                "class_name",
                "train_boxes",
                "train_images",
                "val_boxes",
                "val_images",
                "test_boxes",
                "test_images",
            ],
        )
        writer.writeheader()
        for name, class_id in class_to_id.items():
            writer.writerow(
                {
                    "class_id": class_id,
                    "class_name": name,
                    "train_boxes": counts["train"][name],
                    "train_images": len(image_sets["train"].get(name, set())),
                    "val_boxes": counts["val"][name],
                    "val_images": len(image_sets["val"].get(name, set())),
                    "test_boxes": counts["test"][name],
                    "test_images": len(image_sets["test"].get(name, set())),
                }
            )

    summary = {
        "data_root": str(data_root),
        "output_root": str(output_root),
        "annotation_path": str(annotation_path),
        "annotation_sha256": hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
        "official_train_ids": len(train_pool),
        "official_test_images": len(test_ids),
        "derived_val_images": len(val_ids),
        "derived_train_images": len(train_ids),
        "removed_train_ids_due_to_test_overlap": len(overlap_exclusions),
        "removed_train_images_due_to_exact_content_overlap": len(content_exclusions),
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "classes": len(TYPE45),
        "selected_boxes": {
            split: sum(counts[split].values()) for split in ["train", "val", "test"]
        },
        "link_mode": args.link_mode,
    }
    if args.archive is not None:
        archive = args.archive.resolve()
        if not archive.exists():
            raise FileNotFoundError(archive)
        summary["archive_path"] = str(archive)
        summary["archive_size_bytes"] = archive.stat().st_size
        summary["archive_sha256"] = args.archive_sha256
    (output_root / "manifests" / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    shutil.copy2(stats_path, output_root / "dataset_statistics.csv")
    (output_root / "class_selection.json").write_text(
        json.dumps(
            {
                "source": "official TT100K tutorial anno_func.type45",
                "selection_basis": "45 benchmark classes; selection is frozen before model training",
                "count": len(TYPE45),
                "classes": TYPE45,
                "class_to_id": class_to_id,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
