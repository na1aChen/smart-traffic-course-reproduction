"""Reproducible Stage 3 diagnostics for the locked TT100K baseline.

The script never runs the test split. It optionally creates one validation-set
prediction export with the frozen best.pt, then analyzes either those validation
predictions or the already-frozen test predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
STAGE2_ROOT = PROJECT_ROOT / "02_代码" / "阶段2"
SOURCE_ROOT = PROJECT_ROOT / "02_代码" / "源码" / "yolov10"
DATA_DIR = PROJECT_ROOT / "03_实验" / "datasets" / "TT100K" / "yolo"
DATA_YAML = DATA_DIR / "tt100k.yaml"
BEST_CHECKPOINT = (
    PROJECT_ROOT
    / "03_实验"
    / "baseline_yolov10n"
    / "train"
    / "weights"
    / "best.pt"
)
TEST_EVAL_DIR = PROJECT_ROOT / "03_实验" / "baseline_yolov10n" / "test_eval"
STAGE3_ROOT = PROJECT_ROOT / "03_实验" / "诊断" / "BASELINE-001"
OUTPUT_DIR = STAGE3_ROOT / "analysis"
FIGURE_DIR = OUTPUT_DIR / "figures"
CASE_DIR = OUTPUT_DIR / "cases"
VAL_RUN_DIR = STAGE3_ROOT / "val_eval"

OPERATIONAL_CONF = 0.25
SWEEP_CONFIDENCES = (0.10, 0.15, 0.25, 0.35, 0.50)
MATCH_IOU = 0.50
NEAR_IOU = 0.10


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def ensure_directories() -> None:
    for path in (STAGE3_ROOT, OUTPUT_DIR, FIGURE_DIR, CASE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_names() -> list[str]:
    payload = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    names = payload["names"]
    if not isinstance(names, list) or len(names) != 45:
        raise ValueError(f"Expected 45 class names, got {type(names).__name__}")
    return [str(name) for name in names]


def parse_yolo_file(
    path: Path, width: int, height: int, has_confidence: bool
) -> np.ndarray:
    """Return rows [class_id, x, y, w, h, confidence] in pixel xywh format."""
    if not path.exists():
        return np.empty((0, 6), dtype=np.float32)
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        expected = 6 if has_confidence else 5
        if len(parts) != expected:
            raise ValueError(f"Unexpected label width in {path}: {line!r}")
        class_id = int(float(parts[0]))
        xc, yc, box_w, box_h = map(float, parts[1:5])
        confidence = float(parts[5]) if has_confidence else math.nan
        rows.append(
            [
                class_id,
                xc * width,
                yc * height,
                box_w * width,
                box_h * height,
                confidence,
            ]
        )
    return np.asarray(rows, dtype=np.float32).reshape(-1, 6)


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def load_split(
    split: str,
    gt_dir: Path,
    pred_dir: Path,
    image_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Path]]:
    image_paths = {
        path.stem: path for path in sorted(image_dir.glob("*.jpg"), key=lambda p: p.stem)
    }
    if not image_paths:
        raise FileNotFoundError(f"No images in {image_dir}")
    gt_by_image: dict[str, np.ndarray] = {}
    pred_by_image: dict[str, np.ndarray] = {}
    for index, (image_id, image_path) in enumerate(image_paths.items(), start=1):
        width, height = image_size(image_path)
        gt_by_image[image_id] = parse_yolo_file(
            gt_dir / f"{image_id}.txt", width, height, has_confidence=False
        )
        pred_by_image[image_id] = parse_yolo_file(
            pred_dir / f"{image_id}.txt", width, height, has_confidence=True
        )
        if index % 500 == 0:
            print(f"{split}: loaded {index}/{len(image_paths)} images", flush=True)
    return gt_by_image, pred_by_image, image_paths


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = boxes[:, :4].astype(np.float32, copy=True)
    converted[:, 0] -= converted[:, 2] / 2.0
    converted[:, 1] -= converted[:, 3] / 2.0
    converted[:, 2] += converted[:, 0]
    converted[:, 3] += converted[:, 1]
    return converted


def iou_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.zeros((len(gt_boxes), len(pred_boxes)), dtype=np.float32)
    gt = xywh_to_xyxy(gt_boxes)
    pred = xywh_to_xyxy(pred_boxes)
    left_top = np.maximum(gt[:, None, :2], pred[None, :, :2])
    right_bottom = np.minimum(gt[:, None, 2:], pred[None, :, 2:])
    intersection = np.clip(right_bottom - left_top, 0.0, None)
    inter_area = intersection[:, :, 0] * intersection[:, :, 1]
    gt_area = np.clip(gt[:, 2] - gt[:, 0], 0.0, None) * np.clip(
        gt[:, 3] - gt[:, 1], 0.0, None
    )
    pred_area = np.clip(pred[:, 2] - pred[:, 0], 0.0, None) * np.clip(
        pred[:, 3] - pred[:, 1], 0.0, None
    )
    union = gt_area[:, None] + pred_area[None, :] - inter_area
    return np.divide(
        inter_area,
        union,
        out=np.zeros_like(inter_area, dtype=np.float32),
        where=union > 0,
    )


def match_one_image(
    image_id: str,
    gt: np.ndarray,
    pred: np.ndarray,
    confidence: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pred = pred[pred[:, 5] >= confidence]
    gt_events: list[dict[str, Any]] = []
    pred_events: list[dict[str, Any]] = []
    if len(gt) == 0 and len(pred) == 0:
        return gt_events, pred_events

    overlaps = iou_matrix(gt, pred)
    pair_indices = np.argwhere(overlaps >= MATCH_IOU)
    ordered_pairs = sorted(
        pair_indices,
        key=lambda pair: (
            -float(overlaps[pair[0], pair[1]]),
            -float(pred[pair[1], 5]),
        ),
    )
    gt_to_pred: dict[int, tuple[int, float]] = {}
    pred_to_gt: dict[int, tuple[int, float]] = {}
    for gt_index, pred_index in ordered_pairs:
        gt_index = int(gt_index)
        pred_index = int(pred_index)
        if gt_index in gt_to_pred or pred_index in pred_to_gt:
            continue
        overlap = float(overlaps[gt_index, pred_index])
        gt_to_pred[gt_index] = (pred_index, overlap)
        pred_to_gt[pred_index] = (gt_index, overlap)

    for gt_index, row in enumerate(gt):
        true_class = int(row[0])
        base = {
            "image_id": image_id,
            "true_class_id": true_class,
            "gt_box": [float(value) for value in row[1:5]],
            "gt_area": float(row[3] * row[4]),
            "gt_sqrt_area": float(math.sqrt(max(row[3] * row[4], 0.0))),
        }
        if gt_index in gt_to_pred:
            pred_index, overlap = gt_to_pred[gt_index]
            pred_row = pred[pred_index]
            predicted_class = int(pred_row[0])
            status = "tp" if predicted_class == true_class else "confusion"
            gt_events.append(
                {
                    **base,
                    "status": status,
                    "predicted_class_id": predicted_class,
                    "pred_box": [float(value) for value in pred_row[1:5]],
                    "pred_score": float(pred_row[5]),
                    "best_iou": overlap,
                }
            )
            pred_events.append(
                {
                    "image_id": image_id,
                    "status": "tp" if predicted_class == true_class else "confusion",
                    "true_class_id": true_class,
                    "predicted_class_id": predicted_class,
                    "pred_box": [float(value) for value in pred_row[1:5]],
                    "pred_score": float(pred_row[5]),
                    "best_iou": overlap,
                }
            )
            continue

        unassigned = [
            pred_index
            for pred_index in range(len(pred))
            if pred_index not in pred_to_gt
        ]
        same_class_overlap = 0.0
        any_class_overlap = 0.0
        nearest_pred_index: int | None = None
        for pred_index in unassigned:
            overlap = float(overlaps[gt_index, pred_index])
            any_class_overlap = max(any_class_overlap, overlap)
            if int(pred[pred_index, 0]) == true_class and overlap > same_class_overlap:
                same_class_overlap = overlap
                nearest_pred_index = pred_index
        if same_class_overlap >= NEAR_IOU and nearest_pred_index is not None:
            nearest = pred[nearest_pred_index]
            status = "near_localization"
            extra = {
                "predicted_class_id": true_class,
                "pred_box": [float(value) for value in nearest[1:5]],
                "pred_score": float(nearest[5]),
                "best_iou": same_class_overlap,
            }
        else:
            status = "missed"
            extra = {
                "predicted_class_id": -1,
                "pred_box": [],
                "pred_score": math.nan,
                "best_iou": any_class_overlap,
            }
        gt_events.append({**base, "status": status, **extra})

    for pred_index, row in enumerate(pred):
        if pred_index in pred_to_gt:
            continue
        if len(gt):
            nearest_gt_index = int(np.argmax(overlaps[:, pred_index]))
            nearest_overlap = float(overlaps[nearest_gt_index, pred_index])
            nearest_true_class = int(gt[nearest_gt_index, 0])
        else:
            nearest_overlap = 0.0
            nearest_true_class = -1
        is_duplicate = nearest_overlap >= MATCH_IOU
        pred_events.append(
            {
                "image_id": image_id,
                "status": "duplicate" if is_duplicate else "background_fp",
                "true_class_id": nearest_true_class if is_duplicate else -1,
                "predicted_class_id": int(row[0]),
                "pred_box": [float(value) for value in row[1:5]],
                "pred_score": float(row[5]),
                "best_iou": nearest_overlap,
            }
        )
    return gt_events, pred_events


def collect_events(
    split: str,
    gt_by_image: dict[str, np.ndarray],
    pred_by_image: dict[str, np.ndarray],
    confidence: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gt_records: list[dict[str, Any]] = []
    pred_records: list[dict[str, Any]] = []
    for image_id in sorted(gt_by_image):
        gt_events, pred_events = match_one_image(
            image_id, gt_by_image[image_id], pred_by_image[image_id], confidence
        )
        gt_records.extend(gt_events)
        pred_records.extend(pred_events)
    gt_frame = pd.DataFrame(gt_records)
    pred_frame = pd.DataFrame(pred_records)
    if not gt_frame.empty:
        gt_frame["split"] = split
    if not pred_frame.empty:
        pred_frame["split"] = split
    return gt_frame, pred_frame


def count_summary(gt_frame: pd.DataFrame, pred_frame: pd.DataFrame) -> dict[str, Any]:
    gt_counts = gt_frame["status"].value_counts().to_dict() if len(gt_frame) else {}
    pred_counts = (
        pred_frame["status"].value_counts().to_dict() if len(pred_frame) else {}
    )
    return {
        "ground_truth": int(len(gt_frame)),
        "detections": int(len(pred_frame)),
        "tp": int(gt_counts.get("tp", 0)),
        "confusion": int(gt_counts.get("confusion", 0)),
        "near_localization": int(gt_counts.get("near_localization", 0)),
        "missed": int(gt_counts.get("missed", 0)),
        "background_fp": int(pred_counts.get("background_fp", 0)),
        "duplicate": int(pred_counts.get("duplicate", 0)),
    }


def add_names(
    frame: pd.DataFrame, names: list[str]
) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    if "true_class_id" in frame:
        frame["true_class"] = frame["true_class_id"].map(
            lambda value: names[int(value)] if int(value) >= 0 else "background"
        )
    if "predicted_class_id" in frame:
        frame["predicted_class"] = frame["predicted_class_id"].map(
            lambda value: names[int(value)] if int(value) >= 0 else "none"
        )
    return frame


def build_per_class_table(
    name: str,
    gt_frame: pd.DataFrame,
    pred_frame: pd.DataFrame,
    names: list[str],
    official_metrics: pd.DataFrame,
    class_stats: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    confusion_counts = (
        gt_frame[gt_frame["status"] == "confusion"]
        .groupby(["true_class", "predicted_class"])
        .size()
        if len(gt_frame)
        else pd.Series(dtype=int)
    )
    for class_id, class_name in enumerate(names):
        gt_subset = gt_frame[gt_frame["true_class_id"] == class_id]
        pred_subset = pred_frame[pred_frame["predicted_class_id"] == class_id]
        gt_total = len(gt_subset)
        tp = int((gt_subset["status"] == "tp").sum())
        confusion = int((gt_subset["status"] == "confusion").sum())
        near = int((gt_subset["status"] == "near_localization").sum())
        missed = int((gt_subset["status"] == "missed").sum())
        background = int((pred_subset["status"] == "background_fp").sum())
        duplicate = int((pred_subset["status"] == "duplicate").sum())
        top_confusions: list[str] = []
        if gt_total and len(confusion_counts):
            class_confusions = (
                confusion_counts.xs(class_name, level="true_class")
                if class_name in confusion_counts.index.get_level_values(0)
                else pd.Series(dtype=int)
            )
            for pred_name, count in class_confusions.sort_values(
                ascending=False
            ).items():
                top_confusions.append(f"{pred_name}:{int(count)}")
                if len(top_confusions) == 3:
                    break
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "gt_at_0.25": gt_total,
                "tp_at_0.25": tp,
                "confusion_at_0.25": confusion,
                "near_localization_at_0.25": near,
                "missed_at_0.25": missed,
                "recall_at_0.25": round(tp / gt_total, 6) if gt_total else math.nan,
                "background_fp_at_0.25": background,
                "duplicate_at_0.25": duplicate,
                "top_confused_to": ";".join(top_confusions),
            }
        )
    diagnostics = pd.DataFrame(rows)
    metric_columns = [
        "class_id",
        "class_name",
        "precision",
        "recall",
        "map50",
        "map50_95",
    ]
    diagnostics = diagnostics.merge(
        official_metrics[metric_columns], on=["class_id", "class_name"], how="left"
    )
    support_columns = [
        "class_id",
        "class_name",
        "train_boxes",
        "val_boxes",
        "test_boxes",
    ]
    diagnostics = diagnostics.merge(
        class_stats[support_columns], on=["class_id", "class_name"], how="left"
    )
    diagnostics.insert(0, "split", name)
    return diagnostics


def add_size_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    area = frame["gt_area"].astype(float)
    frame["coco_size"] = pd.cut(
        area,
        bins=[-np.inf, 32**2, 96**2, np.inf],
        labels=["small", "medium", "large"],
        right=False,
    )
    frame["fine_size"] = pd.cut(
        frame["gt_sqrt_area"].astype(float),
        bins=[-np.inf, 16, 24, 32, 48, 64, np.inf],
        labels=["<=16", "16-24", "24-32", "32-48", "48-64", ">64"],
        right=False,
    )
    return frame


def summarize_size_buckets(gt_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bucket_type, column in (("coco", "coco_size"), ("fine", "fine_size")):
        for bucket in gt_frame[column].cat.categories:
            subset = gt_frame[gt_frame[column] == bucket]
            if subset.empty:
                continue
            total = len(subset)
            tp = int((subset["status"] == "tp").sum())
            rows.append(
                {
                    "bucket_type": bucket_type,
                    "bucket": str(bucket),
                    "ground_truth": total,
                    "tp": tp,
                    "confusion": int((subset["status"] == "confusion").sum()),
                    "near_localization": int(
                        (subset["status"] == "near_localization").sum()
                    ),
                    "missed": int((subset["status"] == "missed").sum()),
                    "tp_recall": round(tp / total, 6),
                    "matched_any_recall": round(
                        (
                            (subset["status"] == "tp")
                            | (subset["status"] == "confusion")
                        ).sum()
                        / total,
                        6,
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_confusion_table(gt_frame: pd.DataFrame) -> pd.DataFrame:
    confusion = gt_frame[gt_frame["status"] == "confusion"]
    if confusion.empty:
        return pd.DataFrame(
            columns=[
                "true_class_id",
                "true_class",
                "predicted_class_id",
                "predicted_class",
                "count",
                "share_of_gt",
                "share_of_confusions",
            ]
        )
    table = (
        confusion.groupby(
            [
                "true_class_id",
                "true_class",
                "predicted_class_id",
                "predicted_class",
            ]
        )
        .size()
        .reset_index(name="count")
        .sort_values(["count", "true_class"], ascending=[False, True])
    )
    table["share_of_gt"] = table["count"] / len(gt_frame)
    table["share_of_confusions"] = table["count"] / len(confusion)
    return table


def threshold_sweep(
    split: str,
    gt_by_image: dict[str, np.ndarray],
    pred_by_image: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for confidence in SWEEP_CONFIDENCES:
        gt_frame, pred_frame = collect_events(
            split, gt_by_image, pred_by_image, confidence
        )
        summary = count_summary(gt_frame, pred_frame)
        summary["confidence"] = confidence
        summary["split"] = split
        rows.append(summary)
    return pd.DataFrame(rows)


def image_brightness(path: Path) -> tuple[float, float]:
    with Image.open(path) as image:
        gray = image.convert("L")
        gray.thumbnail((128, 128))
        values = np.asarray(gray, dtype=np.float32)
        return float(values.mean()), float(values.std())


def crop_brightness(path: Path, box: list[float]) -> tuple[float, float]:
    x, y, width, height = box
    with Image.open(path) as image:
        left = max(0, int(x - width))
        top = max(0, int(y - height))
        right = min(image.width, int(x + 2 * width))
        bottom = min(image.height, int(y + 2 * height))
        crop = image.crop((left, top, right, bottom)).convert("L")
        crop.thumbnail((128, 128))
        values = np.asarray(crop, dtype=np.float32)
        return float(values.mean()), float(values.std())


def choose_unique(
    candidates: pd.DataFrame,
    used_images: set[str],
    sort_columns: list[str],
    ascending: list[bool],
) -> pd.Series | None:
    if candidates.empty:
        return None
    ordered = candidates.sort_values(
        sort_columns + ["image_id"], ascending=ascending + [True]
    )
    for _, row in ordered.iterrows():
        if str(row["image_id"]) not in used_images:
            return row
    return ordered.iloc[0] if len(ordered) else None


def add_local_brightness(
    events: pd.DataFrame,
    image_paths: dict[str, Path],
) -> pd.DataFrame:
    """Compute local grayscale mean for each event while opening each image once."""
    if events.empty:
        return events.copy()
    output = events.copy()
    brightness: dict[int, float] = {}
    contrast: dict[int, float] = {}
    for image_id, subset in output.groupby("image_id", sort=False):
        path = image_paths[str(image_id)]
        with Image.open(path) as source:
            gray = source.convert("L")
            for index, row in subset.iterrows():
                x, y, width, height = map(float, row["gt_box"])
                left = max(0, int(x - width))
                top = max(0, int(y - height))
                right = min(gray.width, int(x + 2 * width))
                bottom = min(gray.height, int(y + 2 * height))
                crop = gray.crop((left, top, right, bottom))
                crop.thumbnail((128, 128))
                values = np.asarray(crop, dtype=np.float32)
                brightness[index] = float(values.mean())
                contrast[index] = float(values.std())
    output["local_brightness"] = output.index.map(brightness)
    output["local_contrast"] = output.index.map(contrast)
    return output


def build_case_candidates(
    test_gt: pd.DataFrame,
    test_pred: pd.DataFrame,
    test_per_class: pd.DataFrame,
    image_paths: dict[str, Path],
) -> list[dict[str, Any]]:
    used_images: set[str] = set()
    cases: list[dict[str, Any]] = []

    def append_case(
        title: str,
        rule: str,
        row: pd.Series | None,
        row_type: str,
        full_image: bool = False,
    ) -> None:
        if row is None:
            return
        image_id = str(row["image_id"])
        used_images.add(image_id)
        record = row.to_dict()
        record.update(
            {
                "case_number": len(cases) + 1,
                "title": title,
                "selection_rule": rule,
                "row_type": row_type,
                "full_image": full_image,
                "image_path": str(image_paths[image_id]),
            }
        )
        cases.append(record)

    background = test_pred[test_pred["status"] == "background_fp"]
    append_case(
        "Highest-confidence background false positive",
        "background_fp sorted by descending confidence",
        choose_unique(background, used_images, ["pred_score"], [False]),
        "pred",
    )

    confusion = test_gt[test_gt["status"] == "confusion"]
    append_case(
        "Highest-confidence class confusion",
        "confusion sorted by descending prediction confidence, then IoU",
        choose_unique(confusion, used_images, ["pred_score", "best_iou"], [False, False]),
        "gt",
    )

    small_missed = test_gt[
        (test_gt["status"] == "missed") & (test_gt["gt_area"] <= 32**2)
    ]
    append_case(
        "Small-object miss",
        "small missed GT sorted by ascending area",
        choose_unique(small_missed, used_images, ["gt_area"], [True]),
        "gt",
    )

    total_gt_by_image = test_gt.groupby("image_id").size()
    dense = (
        test_gt[test_gt["status"] == "missed"]
        .groupby("image_id")
        .agg(
            missed_count=("status", "size"),
            lowest_sqrt_area=("gt_sqrt_area", "min"),
        )
        .reset_index()
    )
    dense["gt_count"] = dense["image_id"].map(total_gt_by_image)
    dense = dense.sort_values(
        ["missed_count", "lowest_sqrt_area"], ascending=[False, True]
    )
    dense_row = choose_unique(
        dense, used_images, ["missed_count", "lowest_sqrt_area"], [False, True]
    )
    if dense_row is not None:
        case = dense_row.to_dict()
        case.update(
            {
                "case_number": len(cases) + 1,
                "title": "Dense scene with multiple misses",
                "selection_rule": "image with most missed GT, then smallest missed object",
                "row_type": "image",
                "full_image": True,
                "image_path": str(image_paths[str(case["image_id"])]),
            }
        )
        used_images.add(str(case["image_id"]))
        cases.append(case)

    near = test_gt[test_gt["status"] == "near_localization"]
    append_case(
        "Near-localization failure",
        "near-localization sorted by descending IoU, then confidence",
        choose_unique(near, used_images, ["best_iou", "pred_score"], [False, False]),
        "gt",
    )

    duplicates = test_pred[test_pred["status"] == "duplicate"]
    append_case(
        "Duplicate detection",
        "duplicate sorted by descending confidence",
        choose_unique(duplicates, used_images, ["pred_score"], [False]),
        "pred",
    )

    weakest = set(
        test_per_class.sort_values("map50_95").head(8)["class_name"].tolist()
    )
    rare_missed = test_gt[
        (test_gt["status"] == "missed") & (test_gt["true_class"].isin(weakest))
    ]
    append_case(
        "Weakest-class miss",
        "missed GT in one of the eight lowest mAP50-95 classes, largest first",
        choose_unique(rare_missed, used_images, ["gt_area"], [False]),
        "gt",
    )

    missed = test_gt[
        (test_gt["status"] == "missed") & (test_gt["gt_area"] >= 400)
    ].copy()
    missed = add_local_brightness(missed, image_paths)
    missed = missed.dropna(subset=["local_brightness"])
    append_case(
        "Low-brightness miss",
        "missed GT with the darkest local crop mean; area at least 400 px",
        choose_unique(
            missed,
            used_images,
            ["local_brightness", "gt_area"],
            [True, False],
        ),
        "gt",
    )

    if len(cases) < 8:
        fallback = test_gt[test_gt["status"].isin(["missed", "confusion"])]
        while len(cases) < 8:
            row = choose_unique(fallback, used_images, ["gt_area"], [False])
            if row is None:
                break
            append_case(
                f"Additional deterministic diagnostic case {len(cases) + 1}",
                "largest remaining GT error",
                row,
                "gt",
            )
    return cases[:8]


def get_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
    ):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def draw_labeled_box(
    draw: ImageDraw.ImageDraw,
    box: np.ndarray | list[float],
    color: tuple[int, int, int],
    label: str,
    crop_left: float,
    crop_top: float,
    scale: float,
    header_height: int,
    width: int,
    font: ImageFont.ImageFont,
) -> None:
    x, y, box_width, box_height = box
    left = (x - box_width / 2.0 - crop_left) * scale
    top = header_height + (y - box_height / 2.0 - crop_top) * scale
    right = (x + box_width / 2.0 - crop_left) * scale
    bottom = header_height + (y + box_height / 2.0 - crop_top) * scale
    draw.rectangle((left, top, right, bottom), outline=color, width=width)
    text_box = draw.textbbox((left, top), label, font=font)
    text_height = text_box[3] - text_box[1]
    text_width = text_box[2] - text_box[0]
    label_top = max(header_height, top - text_height - 4)
    draw.rectangle(
        (left, label_top, left + text_width + 4, label_top + text_height + 4),
        fill=(255, 255, 255),
    )
    draw.text((left + 2, label_top + 1), label, fill=color, font=font)


def render_case(
    case: dict[str, Any],
    gt_frame: pd.DataFrame,
    pred_frame: pd.DataFrame,
) -> tuple[Path, dict[str, Any]]:
    image_id = str(case["image_id"])
    image_path = Path(case["image_path"])
    with Image.open(image_path) as source:
        image = source.convert("RGB")

    gt_image = gt_frame[gt_frame["image_id"].astype(str) == image_id]
    pred_image = pred_frame[pred_frame["image_id"].astype(str) == image_id]
    primary_boxes: list[list[float]] = []
    if case.get("gt_box"):
        primary_boxes.append(list(case["gt_box"]))
    if case.get("pred_box"):
        primary_boxes.append(list(case["pred_box"]))
    if case["row_type"] == "image":
        missed_boxes = gt_image[gt_image["status"] == "missed"]["gt_box"].tolist()
        if missed_boxes:
            left = min(box[0] - box[2] / 2 for box in missed_boxes)
            top = min(box[1] - box[3] / 2 for box in missed_boxes)
            right = max(box[0] + box[2] / 2 for box in missed_boxes)
            bottom = max(box[1] + box[3] / 2 for box in missed_boxes)
            margin = max(96.0, 0.15 * max(right - left, bottom - top))
            crop_box = (
                max(0, int(left - margin)),
                max(0, int(top - margin)),
                min(image.width, int(right + margin)),
                min(image.height, int(bottom + margin)),
            )
            source_crop = image.crop(crop_box)
        else:
            crop_box = (0, 0, image.width, image.height)
            source_crop = image.copy()
    else:
        if not primary_boxes:
            crop_box = (0, 0, image.width, image.height)
        else:
            x, y, width, height = primary_boxes[0]
            margin = max(96.0, 2.0 * max(width, height))
            crop_box = (
                max(0, int(x - width / 2 - margin)),
                max(0, int(y - height / 2 - margin)),
                min(image.width, int(x + width / 2 + margin)),
                min(image.height, int(y + height / 2 + margin)),
            )
        source_crop = image.crop(crop_box)

    max_width = 900
    max_height = 1100
    scale = min(4.0, max_width / source_crop.width, max_height / source_crop.height)
    draw_width = max(1, int(source_crop.width * scale))
    draw_height = max(1, int(source_crop.height * scale))
    crop = source_crop.resize((draw_width, draw_height), Image.Resampling.LANCZOS)

    header_height = 72
    canvas = Image.new("RGB", (draw_width, draw_height + header_height), "white")
    canvas.paste(crop, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    title_font = get_font(20)
    box_font = get_font(max(12, int(14 * max(scale, 0.45))))
    draw.text(
        (12, 8),
        f"Case {case['case_number']}: {case['title']}",
        fill=(20, 20, 20),
        font=title_font,
    )
    draw.text(
        (12, 38),
        "green = ground truth, red = prediction",
        fill=(70, 70, 70),
        font=get_font(15),
    )

    for _, row in gt_image.head(40).iterrows():
        center_x, center_y = row["gt_box"][0], row["gt_box"][1]
        if not (
            crop_box[0] <= center_x <= crop_box[2]
            and crop_box[1] <= center_y <= crop_box[3]
        ):
            continue
        primary = bool(case.get("gt_box")) and np.allclose(
            row["gt_box"], case["gt_box"], atol=1e-3
        )
        label = f"GT {row['true_class']}"
        draw_labeled_box(
            draw,
            row["gt_box"],
            (0, 150, 0) if primary else (80, 190, 80),
            label,
            crop_box[0],
            crop_box[1],
            scale,
            header_height,
            max(2, int(4 * max(scale, 0.5))) if primary else 2,
            box_font,
        )
    for _, row in pred_image.head(50).iterrows():
        center_x, center_y = row["pred_box"][0], row["pred_box"][1]
        if not (
            crop_box[0] <= center_x <= crop_box[2]
            and crop_box[1] <= center_y <= crop_box[3]
        ):
            continue
        primary = bool(case.get("pred_box")) and np.allclose(
            row["pred_box"], case["pred_box"], atol=1e-3
        )
        label = f"P {row['predicted_class']} {row['pred_score']:.2f}"
        draw_labeled_box(
            draw,
            row["pred_box"],
            (200, 0, 0) if primary else (235, 110, 110),
            label,
            crop_box[0],
            crop_box[1],
            scale,
            header_height,
            max(2, int(4 * max(scale, 0.5))) if primary else 2,
            box_font,
        )

    output_path = CASE_DIR / f"case_{case['case_number']:02d}_{image_id}.png"
    canvas.save(output_path)
    primary_gt_box = list(case["gt_box"]) if case.get("gt_box") else []
    brightness = (
        crop_brightness(image_path, primary_gt_box)[0] if primary_gt_box else math.nan
    )
    return output_path, {
        "brightness_mean": brightness,
        "crop_width": crop_box[2] - crop_box[0],
        "crop_height": crop_box[3] - crop_box[1],
    }


def build_case_manifest(
    cases: list[dict[str, Any]],
    images: dict[str, Path],
    gt_frame: pd.DataFrame,
    pred_frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case in cases:
        output_path, metadata = render_case(case, gt_frame, pred_frame)
        image_id = str(case["image_id"])
        dense_gt = gt_frame[gt_frame["image_id"].astype(str) == image_id]
        dense_pred = pred_frame[pred_frame["image_id"].astype(str) == image_id]
        rows.append(
            {
                "case_number": case["case_number"],
                "case_type": case["title"],
                "selection_rule": case["selection_rule"],
                "image_id": image_id,
                "image_path": str(images[image_id]),
                "true_class": case.get("true_class", ""),
                "predicted_class": case.get("predicted_class", ""),
                "status": case.get("status", "image"),
                "pred_score": case.get("pred_score", math.nan),
                "best_iou": case.get("best_iou", math.nan),
                "gt_area": case.get("gt_area", math.nan),
                "image_missed_count": int((dense_gt["status"] == "missed").sum()),
                "image_gt_count": int(len(dense_gt)),
                "image_detection_count_at_0.25": int(len(dense_pred)),
                "local_brightness": metadata["brightness_mean"],
                "crop_width": metadata["crop_width"],
                "crop_height": metadata["crop_height"],
                "output_path": str(output_path),
            }
        )
    return pd.DataFrame(rows)


def build_montage(manifest: pd.DataFrame) -> Path:
    cards: list[Image.Image] = []
    for _, row in manifest.iterrows():
        with Image.open(Path(row["output_path"])) as image:
            card = image.convert("RGB")
            card.thumbnail((500, 360))
            base = Image.new("RGB", (500, 360), "white")
            base.paste(card, ((500 - card.width) // 2, (360 - card.height) // 2))
            cards.append(base)
    columns = 2
    rows = math.ceil(len(cards) / columns)
    montage = Image.new("RGB", (columns * 500, rows * 360), "white")
    for index, card in enumerate(cards):
        x = (index % columns) * 500
        y = (index // columns) * 360
        montage.paste(card, (x, y))
    output_path = FIGURE_DIR / "case_montage.png"
    montage.save(output_path)
    return output_path


def make_figures(
    per_class: pd.DataFrame,
    size_table: pd.DataFrame,
    confusion: pd.DataFrame,
    sweep: pd.DataFrame,
) -> list[Path]:
    outputs: list[Path] = []

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        per_class["train_boxes"],
        per_class["map50_95"],
        c=per_class["test_boxes"],
        cmap="viridis",
        s=42,
        alpha=0.85,
    )
    for _, row in per_class.sort_values("map50_95").head(8).iterrows():
        ax.annotate(
            row["class_name"],
            (row["train_boxes"], row["map50_95"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Training boxes (log scale)")
    ax.set_ylabel("Test mAP50-95")
    ax.set_title("Class AP versus training support")
    fig.colorbar(scatter, ax=ax, label="Test boxes")
    fig.tight_layout()
    path = FIGURE_DIR / "class_ap_vs_support.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    coco = size_table[size_table["bucket_type"] == "coco"].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(coco["bucket"], coco["tp_recall"], color=["#c94c4c", "#e6a23c", "#4c9a62"])
    for index, row in coco.reset_index(drop=True).iterrows():
        ax.text(
            index,
            row["tp_recall"] + 0.02,
            f"{row['tp_recall']:.3f}\n(n={int(row['ground_truth'])})",
            ha="center",
            fontsize=9,
        )
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"TP recall at confidence {OPERATIONAL_CONF:.2f}")
    ax.set_title("Ground-truth recall by COCO size bucket")
    fig.tight_layout()
    path = FIGURE_DIR / "size_bucket_recall.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    top = confusion.head(15).sort_values("count")
    fig, ax = plt.subplots(figsize=(9, 6))
    labels = [
        f"{row.true_class} -> {row.predicted_class}"
        for row in top.itertuples()
    ]
    ax.barh(labels, top["count"], color="#b24a56")
    ax.set_xlabel("Confused ground-truth instances at IoU >= 0.50")
    ax.set_title("Top class-confusion directions")
    fig.tight_layout()
    path = FIGURE_DIR / "top_confusion_pairs.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep["confidence"], sweep["tp"], marker="o", label="TP")
    ax.plot(sweep["confidence"], sweep["confusion"], marker="o", label="wrong class")
    ax.plot(sweep["confidence"], sweep["missed"], marker="o", label="missed")
    ax.plot(
        sweep["confidence"],
        sweep["background_fp"],
        marker="o",
        label="background FP",
    )
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Count")
    ax.set_title("Operational error counts across thresholds")
    ax.legend()
    fig.tight_layout()
    path = FIGURE_DIR / "threshold_sweep.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)
    return outputs


def markdown_table(rows: list[list[Any]], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |" for row in rows
    )
    return "\n".join(lines)


def mechanism_snapshot(
    split: str,
    size_table: pd.DataFrame,
    confusion: pd.DataFrame,
    gt_frame: pd.DataFrame,
    per_class: pd.DataFrame,
) -> dict[str, Any]:
    coco = size_table[size_table["bucket_type"] == "coco"].set_index("bucket")
    small = coco.loc["small"].to_dict() if "small" in coco.index else {}
    medium = coco.loc["medium"].to_dict() if "medium" in coco.index else {}
    large = coco.loc["large"].to_dict() if "large" in coco.index else {}
    top_confusions = confusion.head(5).to_dict(orient="records")
    speed_pairs = 0
    for row in confusion.itertuples():
        if row.true_class.startswith(("pl", "pm")) and row.predicted_class.startswith(
            ("pl", "pm")
        ):
            speed_pairs += int(row.count)
    size_counts = gt_frame["coco_size"].value_counts()
    missed_counts = gt_frame[gt_frame["status"] == "missed"]["coco_size"].value_counts()
    within_class_gaps: list[float] = []
    for _, subset in gt_frame.groupby("true_class"):
        small_subset = subset[subset["coco_size"] == "small"]
        medium_subset = subset[subset["coco_size"] == "medium"]
        if len(small_subset) < 5 or len(medium_subset) < 5:
            continue
        small_recall = float((small_subset["status"] == "tp").mean())
        medium_recall = float((medium_subset["status"] == "tp").mean())
        within_class_gaps.append(medium_recall - small_recall)
    support_correlation = per_class[["train_boxes", "map50_95"]].corr(
        method="spearman"
    ).iloc[0, 1]
    return {
        "split": split,
        "ground_truth_by_size": {
            str(key): int(value) for key, value in size_counts.items()
        },
        "missed_by_size": {
            str(key): int(value) for key, value in missed_counts.items()
        },
        "small_ground_truth_share": (
            float(size_counts.get("small", 0) / len(gt_frame)) if len(gt_frame) else None
        ),
        "small_share_of_all_misses": (
            float(missed_counts.get("small", 0) / missed_counts.sum())
            if missed_counts.sum()
            else None
        ),
        "small_tp_recall": small.get("tp_recall"),
        "medium_tp_recall": medium.get("tp_recall"),
        "large_tp_recall": large.get("tp_recall"),
        "small_minus_medium_recall": (
            float(small["tp_recall"]) - float(medium["tp_recall"])
            if small and medium
            else None
        ),
        "confusion_count": int((gt_frame["status"] == "confusion").sum()),
        "confusion_share_of_gt": float(
            (gt_frame["status"] == "confusion").mean()
        ),
        "speed_limit_family_confusions": speed_pairs,
        "speed_limit_family_share_of_confusions": (
            speed_pairs / int((gt_frame["status"] == "confusion").sum())
            if int((gt_frame["status"] == "confusion").sum())
            else None
        ),
        "within_class_size_control": {
            "classes_with_at_least_5_small_and_5_medium": len(within_class_gaps),
            "classes_with_medium_recall_above_small": sum(
                gap > 0 for gap in within_class_gaps
            ),
            "median_medium_minus_small_recall": (
                float(np.median(within_class_gaps)) if within_class_gaps else None
            ),
        },
        "top_confusions": top_confusions,
        "spearman_train_support_vs_map50_95": (
            None if math.isnan(float(support_correlation)) else float(support_correlation)
        ),
    }


def write_analysis_tables(
    per_class: pd.DataFrame,
    size_table: pd.DataFrame,
    confusion: pd.DataFrame,
    sweep: pd.DataFrame,
    manifest: pd.DataFrame,
    summary: dict[str, Any],
) -> Path:
    worst = per_class.sort_values(["map50_95", "class_name"]).head(12)
    class_rows = [
        [
            row["class_name"],
            int(row["test_boxes"]),
            f"{row['precision']:.4f}",
            f"{row['recall']:.4f}",
            f"{row['map50']:.4f}",
            f"{row['map50_95']:.4f}",
            f"{row['recall_at_0.25']:.4f}",
            int(row["missed_at_0.25"]),
            int(row["confusion_at_0.25"]),
        ]
        for row in worst.to_dict(orient="records")
    ]
    all_class_rows = [
        [
            int(row["class_id"]),
            row["class_name"],
            int(row["test_boxes"]),
            f"{row['precision']:.4f}",
            f"{row['recall']:.4f}",
            f"{row['map50']:.4f}",
            f"{row['map50_95']:.4f}",
            f"{row['recall_at_0.25']:.4f}",
            int(row["missed_at_0.25"]),
            int(row["confusion_at_0.25"]),
            row["top_confused_to"],
        ]
        for row in per_class.sort_values("class_id").to_dict(orient="records")
    ]
    coco = size_table[size_table["bucket_type"] == "coco"]
    size_rows = [
        [
            row.bucket,
            int(row.ground_truth),
            int(row.tp),
            int(row.confusion),
            int(row.near_localization),
            int(row.missed),
            f"{row.tp_recall:.4f}",
        ]
        for row in coco.itertuples()
    ]
    confusion_rows = [
        [
            row.true_class,
            row.predicted_class,
            int(row.count),
            f"{row.share_of_gt:.4f}",
        ]
        for row in confusion.head(15).itertuples()
    ]
    case_rows = [
        [
            int(row.case_number),
            row.case_type,
            row.image_id,
            row.true_class,
            row.predicted_class,
            f"{row.pred_score:.4f}"
            if not math.isnan(float(row.pred_score))
            else "",
        ]
        for row in manifest.itertuples()
    ]
    content = [
        "# 阶段 3 自动生成诊断表",
        "",
        f"生成时间：{summary['created_at']}",
        "",
        "## 最弱类别",
        "",
        markdown_table(
            class_rows,
            [
                "类别",
                "test框",
                "P",
                "R",
                "mAP50",
                "mAP50-95",
                "0.25召回",
                "漏检",
                "错类",
            ],
        ),
        "",
        "## 全部 45 类指标",
        "",
        markdown_table(
            all_class_rows,
            [
                "ID",
                "类别",
                "test框",
                "P",
                "R",
                "mAP50",
                "mAP50-95",
                "0.25召回",
                "漏检",
                "错类",
                "主要误判方向",
            ],
        ),
        "",
        "## COCO 尺寸分桶",
        "",
        markdown_table(
            size_rows,
            ["尺寸", "真实框", "TP", "错类", "近定位", "漏检", "阈值召回"],
        ),
        "",
        "## 主要混淆对",
        "",
        markdown_table(
            confusion_rows,
            ["真实类", "预测类", "数量", "占全部真实框"],
        ),
        "",
        "## 典型案例",
        "",
        markdown_table(
            case_rows,
            ["编号", "类型", "图像", "真实类", "预测类", "置信度"],
        ),
        "",
    ]
    output_path = OUTPUT_DIR / "analysis_tables.md"
    output_path.write_text("\n".join(content), encoding="utf-8")
    return output_path


def prepare_val() -> int:
    manifest_path = VAL_RUN_DIR / "stage3_val_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") == "passed":
            print(f"Validation prediction export already complete: {VAL_RUN_DIR}")
            return 0
    ensure_directories()
    sys.path.insert(0, str(STAGE2_ROOT))
    sys.path.insert(0, str(SOURCE_ROOT))
    from stage2_runner import YOLOv10, install_unicode_path_save_patch, tee_console

    install_unicode_path_save_patch()
    VAL_RUN_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLOv10(str(BEST_CHECKPOINT))
    with tee_console(VAL_RUN_DIR / "console.log"):
        metrics = model.val(
            data=str(DATA_YAML),
            split="val",
            imgsz=640,
            batch=8,
            device="0",
            workers=0,
            project=str(STAGE3_ROOT),
            name="val_eval",
            exist_ok=True,
            save_json=False,
            save_txt=True,
            save_conf=True,
            plots=True,
            conf=0.001,
            iou=0.7,
            max_det=100,
            seed=2026,
            deterministic=True,
            verbose=True,
        )
    payload = {
        "created_at": utc_now(),
        "status": "passed",
        "purpose": "Validation-only diagnostic confirmation; test split was not run",
        "checkpoint": str(BEST_CHECKPOINT),
        "data_yaml": str(DATA_YAML),
        "split": "val",
        "conf": 0.001,
        "iou": 0.7,
        "max_det": 100,
        "results": {str(key): float(value) for key, value in metrics.results_dict.items()},
        "speed_ms_per_image": {
            str(key): float(value) for key, value in metrics.speed.items()
        },
        "label_directory": str(VAL_RUN_DIR / "labels"),
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Validation diagnostic export: {manifest_path}")
    return 0


def analyze() -> int:
    ensure_directories()
    names = load_names()
    official_metrics = pd.read_csv(TEST_EVAL_DIR / "per_class_metrics.csv")
    class_stats = pd.read_csv(DATA_DIR / "stats" / "class_statistics.csv")
    test_gt, test_pred, test_images = load_split(
        "test",
        DATA_DIR / "labels" / "test",
        TEST_EVAL_DIR / "labels",
        DATA_DIR / "images" / "test",
    )
    val_manifest = VAL_RUN_DIR / "stage3_val_manifest.json"
    if not val_manifest.exists():
        raise FileNotFoundError(
            "Validation diagnostic export is missing; run --prepare-val first"
        )
    val_gt, val_pred, _ = load_split(
        "val",
        DATA_DIR / "labels" / "val",
        VAL_RUN_DIR / "labels",
        DATA_DIR / "images" / "val",
    )

    coco = json.loads(
        (TEST_EVAL_DIR / "ground_truth_coco.json").read_text(encoding="utf-8")
    )
    test_gt_count = sum(len(values) for values in test_gt.values())
    test_pred_count = sum(len(values) for values in test_pred.values())
    frozen_check = {
        "test_images_loaded_from_frozen_split": len(test_images),
        "test_ground_truth": test_gt_count,
        "test_predictions_loaded_from_saved_labels": test_pred_count,
        "coco_images": len(coco["images"]),
        "coco_annotations": len(coco["annotations"]),
        "coco_categories": len(coco["categories"]),
        "test_inference_rerun": False,
        "consistent": (
            len(test_images) == len(coco["images"]) == 3071
            and test_gt_count == len(coco["annotations"]) == 7706
            and test_pred_count == 307100
        ),
    }
    if not frozen_check["consistent"]:
        raise ValueError(f"Frozen test evidence check failed: {frozen_check}")

    test_gt_events, test_pred_events = collect_events(
        "test", test_gt, test_pred, OPERATIONAL_CONF
    )
    val_gt_events, val_pred_events = collect_events(
        "val", val_gt, val_pred, OPERATIONAL_CONF
    )
    test_gt_events = add_names(add_size_buckets(test_gt_events), names)
    val_gt_events = add_names(add_size_buckets(val_gt_events), names)
    test_pred_events = add_names(test_pred_events, names)
    val_pred_events = add_names(val_pred_events, names)

    test_per_class = build_per_class_table(
        "test",
        test_gt_events,
        test_pred_events,
        names,
        official_metrics,
        class_stats,
    )
    val_official_metrics = official_metrics.copy()
    for column in ("precision", "recall", "map50", "map50_95"):
        val_official_metrics[column] = math.nan
    val_per_class = build_per_class_table(
        "val",
        val_gt_events,
        val_pred_events,
        names,
        val_official_metrics,
        class_stats,
    )
    per_class = pd.concat([test_per_class, val_per_class], ignore_index=True)
    test_size = summarize_size_buckets(test_gt_events)
    test_size.insert(0, "split", "test")
    val_size = summarize_size_buckets(val_gt_events)
    val_size.insert(0, "split", "val")
    size_table = pd.concat([test_size, val_size], ignore_index=True)
    test_confusion = build_confusion_table(test_gt_events)
    test_confusion.insert(0, "split", "test")
    val_confusion = build_confusion_table(val_gt_events)
    val_confusion.insert(0, "split", "val")
    confusion_table = pd.concat([test_confusion, val_confusion], ignore_index=True)
    test_sweep = threshold_sweep("test", test_gt, test_pred)
    val_sweep = threshold_sweep("val", val_gt, val_pred)
    sweep = pd.concat([test_sweep, val_sweep], ignore_index=True)

    per_class.to_csv(
        OUTPUT_DIR / "per_class_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    size_table.to_csv(
        OUTPUT_DIR / "size_bucket_metrics.csv", index=False, encoding="utf-8-sig"
    )
    confusion_table.to_csv(
        OUTPUT_DIR / "confusion_pairs.csv", index=False, encoding="utf-8-sig"
    )
    sweep.to_csv(
        OUTPUT_DIR / "threshold_sweep.csv", index=False, encoding="utf-8-sig"
    )
    test_gt_events.to_csv(
        OUTPUT_DIR / "test_gt_events.csv", index=False, encoding="utf-8-sig"
    )
    test_pred_events.to_csv(
        OUTPUT_DIR / "test_pred_events.csv", index=False, encoding="utf-8-sig"
    )
    val_gt_events.to_csv(
        OUTPUT_DIR / "val_gt_events.csv", index=False, encoding="utf-8-sig"
    )
    val_pred_events.to_csv(
        OUTPUT_DIR / "val_pred_events.csv", index=False, encoding="utf-8-sig"
    )

    test_confusion_only = test_confusion.drop(columns=["split"])
    cases = build_case_candidates(
        test_gt_events, test_pred_events, test_per_class, test_images
    )
    case_manifest = build_case_manifest(
        cases, test_images, test_gt_events, test_pred_events
    )
    montage_path = build_montage(case_manifest)
    case_manifest.to_csv(
        OUTPUT_DIR / "case_manifest.csv", index=False, encoding="utf-8-sig"
    )

    figure_paths = make_figures(
        test_per_class, test_size, test_confusion_only, test_sweep
    )
    figure_paths.append(montage_path)
    summary = {
        "created_at": utc_now(),
        "experiment_id": "BASELINE-001",
        "purpose": "Stage 3 descriptive error diagnosis and validation confirmation",
        "operational_confidence": OPERATIONAL_CONF,
        "match_iou": MATCH_IOU,
        "near_iou": NEAR_IOU,
        "frozen_test_check": frozen_check,
        "test_summary": count_summary(test_gt_events, test_pred_events),
        "val_summary": count_summary(val_gt_events, val_pred_events),
        "mechanism_snapshot": {
            "test": mechanism_snapshot(
                "test",
                test_size,
                test_confusion_only,
                test_gt_events,
                test_per_class,
            ),
            "val": mechanism_snapshot(
                "val",
                val_size,
                val_confusion.drop(columns=["split"]),
                val_gt_events,
                val_per_class,
            ),
        },
        "outputs": {
            "per_class": str(OUTPUT_DIR / "per_class_diagnostics.csv"),
            "size_bucket": str(OUTPUT_DIR / "size_bucket_metrics.csv"),
            "confusion": str(OUTPUT_DIR / "confusion_pairs.csv"),
            "threshold_sweep": str(OUTPUT_DIR / "threshold_sweep.csv"),
            "test_gt_events": str(OUTPUT_DIR / "test_gt_events.csv"),
            "test_pred_events": str(OUTPUT_DIR / "test_pred_events.csv"),
            "val_gt_events": str(OUTPUT_DIR / "val_gt_events.csv"),
            "val_pred_events": str(OUTPUT_DIR / "val_pred_events.csv"),
            "case_manifest": str(OUTPUT_DIR / "case_manifest.csv"),
            "figures": [str(path) for path in figure_paths],
        },
    }
    (OUTPUT_DIR / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_analysis_tables(
        test_per_class,
        test_size,
        test_confusion_only,
        test_sweep,
        case_manifest,
        summary,
    )
    print(json.dumps(summary["test_summary"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["val_summary"], ensure_ascii=False, indent=2))
    print(f"Stage 3 analysis: {OUTPUT_DIR}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-val", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.prepare_val:
        return prepare_val()
    if args.analyze or not any((args.prepare_val, args.analyze)):
        return analyze()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
