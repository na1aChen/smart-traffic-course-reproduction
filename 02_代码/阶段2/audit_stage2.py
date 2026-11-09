"""Consolidate Stage 2 artifacts and verify the baseline evidence chain."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SOURCE_ROOT = PROJECT_ROOT / "02_代码" / "源码" / "yolov10"
EXP_ROOT = PROJECT_ROOT / "03_实验" / "baseline_yolov10n"
TRAIN_DIR = EXP_ROOT / "train"
EVAL_DIR = EXP_ROOT / "test_eval"
EVIDENCE_ROOT = PROJECT_ROOT / "01_立项" / "阶段2_证据"
DATASET_ROOT = PROJECT_ROOT / "03_实验" / "datasets" / "TT100K" / "yolo"

TRAIN_SUMMARY = EVIDENCE_ROOT / "stage2_training_summary.json"
EVAL_SUMMARY = EVIDENCE_ROOT / "stage2_test_evaluation.json"
WEIGHT_EQUIVALENCE = EVIDENCE_ROOT / "权重等价性校验.json"
COCO_ROUNDTRIP = EXP_ROOT / "qa_coco_roundtrip" / "qa_coco_roundtrip.json"
RESULTS_CSV = TRAIN_DIR / "results.csv"
TRAIN_ARGS = TRAIN_DIR / "args.yaml"
BEST_CHECKPOINT = TRAIN_DIR / "weights" / "best.pt"
LAST_CHECKPOINT = TRAIN_DIR / "weights" / "last.pt"
PER_CLASS_CSV = EVAL_DIR / "per_class_metrics.csv"
YOLO_METRICS_JSON = EVAL_DIR / "metrics_yolov10.json"
PREDICTIONS_JSON = EVAL_DIR / "predictions_category0.json"
GROUND_TRUTH_JSON = EVAL_DIR / "ground_truth_coco.json"
AUDIT_JSON = EVIDENCE_ROOT / "阶段2最终审计.json"
PROBE_SUMMARY = EVIDENCE_ROOT / "stage2_probe_summary.json"
COMMAND_COMPARISON = SCRIPT_DIR / "官方命令与本机命令对照.md"
EXPERIMENT_DESIGN = EVIDENCE_ROOT / "阶段2实验设计.md"

sys.path.insert(0, str(SOURCE_ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV while trimming padding from Ultralytics-style headers."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            headers = [header.strip() for header in next(reader)]
        except StopIteration:
            return []
        return [
            {header: value.strip() for header, value in zip(headers, row, strict=False)}
            for row in reader
        ]


def file_evidence(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256(path) if path.is_file() else None,
    }


def main() -> int:
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, passed: bool, detail: Any) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    required_paths = [
        TRAIN_SUMMARY,
        EVAL_SUMMARY,
        WEIGHT_EQUIVALENCE,
        COCO_ROUNDTRIP,
        RESULTS_CSV,
        TRAIN_ARGS,
        BEST_CHECKPOINT,
        LAST_CHECKPOINT,
        PER_CLASS_CSV,
        YOLO_METRICS_JSON,
        PREDICTIONS_JSON,
        GROUND_TRUTH_JSON,
        PROBE_SUMMARY,
        COMMAND_COMPARISON,
        EXPERIMENT_DESIGN,
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    check("required_artifacts_exist", not missing, {"missing": missing})
    if missing:
        payload = {
            "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "failed",
            "checks": checks,
        }
        AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
        AUDIT_JSON.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1

    train_summary = read_json(TRAIN_SUMMARY)
    eval_summary = read_json(EVAL_SUMMARY)
    weight_equivalence = read_json(WEIGHT_EQUIVALENCE)
    coco_roundtrip = read_json(COCO_ROUNDTRIP)
    yolo_metrics = read_json(YOLO_METRICS_JSON)
    probe_summary = read_json(PROBE_SUMMARY)

    check(
        "training_status_passed",
        train_summary.get("status") == "passed",
        train_summary.get("status"),
    )
    check(
        "evaluation_status_passed",
        eval_summary.get("status") == "passed",
        eval_summary.get("status"),
    )
    check(
        "weights_exactly_equivalent",
        weight_equivalence.get("all_tensors_exactly_equal") is True
        and weight_equivalence.get("mismatch_count") == 0,
        {
            "tensor_count": weight_equivalence.get("tensor_count_hf"),
            "mismatch_count": weight_equivalence.get("mismatch_count"),
        },
    )
    check(
        "coco_roundtrip_passed",
        coco_roundtrip.get("passed") is True,
        coco_roundtrip.get("metrics"),
    )
    passed_probes = [
        item.get("run_name")
        for item in probe_summary.get("results", [])
        if item.get("status") == "passed"
    ]
    check(
        "data_entry_probe_passed",
        bool(passed_probes),
        {"passed_probes": passed_probes},
    )

    train_args = yaml.safe_load(TRAIN_ARGS.read_text(encoding="utf-8"))
    locked_config = {
        "seed": train_args.get("seed"),
        "imgsz": train_args.get("imgsz"),
        "batch": train_args.get("batch"),
        "optimizer": train_args.get("optimizer"),
        "amp": train_args.get("amp"),
        "deterministic": train_args.get("deterministic"),
        "split": train_args.get("split"),
    }
    check(
        "baseline_config_locked",
        locked_config
        == {
            "seed": 2026,
            "imgsz": 640,
            "batch": 8,
            "optimizer": "SGD",
            "amp": True,
            "deterministic": True,
            "split": "val",
        },
        locked_config,
    )

    result_rows = read_csv_rows(RESULTS_CSV)
    numeric_rows: list[dict[str, float]] = []
    all_finite = True
    for row in result_rows:
        converted: dict[str, float] = {}
        for key, value in row.items():
            if key is None or value in (None, ""):
                continue
            try:
                converted[key] = float(value)
            except ValueError:
                continue
        numeric_rows.append(converted)
        all_finite = all_finite and all(
            math.isfinite(value) for value in converted.values()
        )
    for row in numeric_rows:
        # YOLOv10's DetectionMetrics fitness weights mAP50 by 0.1 and
        # mAP50-95 by 0.9. The compact results.csv omits the fitness column.
        row["fitness_derived"] = (
            0.1 * row.get("metrics/mAP50(B)", 0.0)
            + 0.9 * row.get("metrics/mAP50-95(B)", 0.0)
        )
    check(
        "results_csv_present_and_finite",
        bool(result_rows) and all_finite,
        {"epochs_recorded": len(result_rows), "all_numeric_values_finite": all_finite},
    )

    fitness_key = "fitness_derived"
    map_key = "metrics/mAP50-95(B)"
    best_row = max(numeric_rows, key=lambda row: row.get(fitness_key, -math.inf))
    check(
        "validation_best_recorded",
        math.isfinite(best_row.get(fitness_key, -math.inf)),
        {
            "epoch": int(best_row["epoch"]),
            "fitness": best_row.get(fitness_key),
            "map50_95": best_row.get(map_key),
            "map50": best_row.get("metrics/mAP50(B)"),
        },
    )

    checkpoint = torch.load(BEST_CHECKPOINT, map_location="cpu")
    raw_checkpoint_epoch = checkpoint.get("epoch")
    checkpoint_epoch = (
        int(raw_checkpoint_epoch) if raw_checkpoint_epoch is not None else None
    )
    checkpoint_epoch_1based = (
        int(checkpoint_epoch) + 1
        if checkpoint_epoch is not None and checkpoint_epoch >= 0
        else None
    )
    checkpoint_fitness = checkpoint.get("train_metrics", {}).get("fitness")
    if checkpoint_fitness is None:
        checkpoint_fitness = checkpoint.get("best_fitness")
    checkpoint_fitness = (
        float(checkpoint_fitness) if checkpoint_fitness is not None else None
    )
    check(
        "best_checkpoint_loads",
        "model" in checkpoint and checkpoint_epoch is not None,
        {
            "raw_epoch_after_strip": checkpoint_epoch,
            "epoch_one_based_before_strip_if_available": checkpoint_epoch_1based,
            "fitness": checkpoint_fitness,
            "keys": sorted(checkpoint.keys()),
        },
    )
    check(
        "best_checkpoint_matches_csv",
        checkpoint_fitness is not None
        and abs(float(checkpoint_fitness) - best_row[fitness_key]) < 5e-5,
        {
            "raw_checkpoint_epoch_after_strip": checkpoint_epoch,
            "checkpoint_epoch_one_based_if_available": checkpoint_epoch_1based,
            "checkpoint_epoch_note": (
                "Ultralytics strip_optimizer rewrites the final best.pt epoch to -1; "
                "identity is verified by train_metrics fitness and metric values."
            ),
            "checkpoint_fitness": checkpoint_fitness,
            "csv_best_epoch": int(best_row["epoch"]),
            "csv_best_fitness_derived": best_row[fitness_key],
            "comparison_tolerance": 5e-5,
        },
    )

    split_records = read_json(DATASET_ROOT / "manifests" / "split_manifest.json")
    ids_by_split: dict[str, set[str]] = defaultdict(set)
    hashes_by_split: dict[str, set[str]] = defaultdict(set)
    for record in split_records:
        ids_by_split[str(record["split"])].add(str(record["image_id"]))
        hashes_by_split[str(record["split"])].add(str(record["image_sha256"]))
    overlap_details: dict[str, int] = {}
    split_names = ["train", "val", "test"]
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            overlap_details[f"{left}_{right}_ids"] = len(
                ids_by_split[left] & ids_by_split[right]
            )
            overlap_details[f"{left}_{right}_content_hashes"] = len(
                hashes_by_split[left] & hashes_by_split[right]
            )
    check(
        "splits_disjoint",
        all(value == 0 for value in overlap_details.values()),
        overlap_details,
    )

    ground_truth = read_json(GROUND_TRUTH_JSON)
    predictions = read_json(PREDICTIONS_JSON)
    prediction_files = sorted((EVAL_DIR / "labels").glob("*.txt"))
    check(
        "test_ground_truth_complete",
        len(ground_truth.get("images", [])) == 3071
        and len(ground_truth.get("categories", [])) == 45
        and len(ground_truth.get("annotations", [])) == 7706,
        {
            "images": len(ground_truth.get("images", [])),
            "categories": len(ground_truth.get("categories", [])),
            "annotations": len(ground_truth.get("annotations", [])),
        },
    )
    check(
        "raw_predictions_saved",
        bool(predictions) and len(prediction_files) > 0,
        {
            "prediction_count": len(predictions),
            "prediction_files": len(prediction_files),
        },
    )

    per_class_rows = read_csv_rows(PER_CLASS_CSV)
    check(
        "per_class_metrics_present",
        0 < len(per_class_rows) <= 45,
        {"class_rows": len(per_class_rows)},
    )

    recomputed = eval_summary.get("recomputed_metrics", {})
    summary_yolo_metrics = eval_summary.get("metrics_yolov10", {})
    yolo_map50 = float(yolo_metrics["metrics/mAP50(B)"])
    yolo_map50_95 = float(yolo_metrics["metrics/mAP50-95(B)"])
    coco_map50 = float(recomputed["map50"])
    coco_map50_95 = float(recomputed["map50_95"])
    map50_difference = abs(yolo_map50 - coco_map50)
    map50_95_difference = abs(yolo_map50_95 - coco_map50_95)
    check(
        "independent_metric_recomputed",
        all(
            math.isfinite(value)
            for value in [yolo_map50, yolo_map50_95, coco_map50, coco_map50_95]
        )
        and map50_difference <= 0.01
        and map50_95_difference <= 0.01,
        {
            "yolov10_map50": yolo_map50,
            "yolov10_map50_95": yolo_map50_95,
            "coco_map50": coco_map50,
            "coco_map50_95": coco_map50_95,
            "map50_absolute_difference": map50_difference,
            "map50_95_absolute_difference": map50_95_difference,
            "max_allowed_absolute_difference": 0.01,
        },
    )

    plot_names = [
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "val_batch0_labels.jpg",
        "val_batch0_pred.jpg",
    ]
    plot_evidence = {
        name: (TRAIN_DIR / name).is_file()
        for name in plot_names
    }
    check(
        "training_curves_and_validation_plots_present",
        all(plot_evidence.values()),
        plot_evidence,
    )

    important_files = [
        TRAIN_SUMMARY,
        EVAL_SUMMARY,
        WEIGHT_EQUIVALENCE,
        COCO_ROUNDTRIP,
        RESULTS_CSV,
        BEST_CHECKPOINT,
        LAST_CHECKPOINT,
        PER_CLASS_CSV,
        YOLO_METRICS_JSON,
        PREDICTIONS_JSON,
        GROUND_TRUTH_JSON,
        PROBE_SUMMARY,
        COMMAND_COMPARISON,
        EXPERIMENT_DESIGN,
        EVIDENCE_ROOT / "环境补充记录.md",
        EVIDENCE_ROOT / "恢复与失败记录.md",
        EVIDENCE_ROOT / "图表目视QA.md",
        EVIDENCE_ROOT / "stage2_training_summary.partial_invalid.json",
        EVIDENCE_ROOT / "stage2_training_summary.recovered.json",
        EVIDENCE_ROOT / "stage2_test_evaluation.initial_failure.json",
        EVIDENCE_ROOT / "stage2_test_evaluation.recovered.json",
        TRAIN_DIR / "args.yaml",
        TRAIN_DIR / "console.log",
    ]
    status = "passed" if all(item["passed"] for item in checks.values()) else "failed"
    payload = {
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment_id": "BASE_YOLOV10N_TT100K45_S2026",
        "status": status,
        "summary_metrics": {
            "best_epoch_csv": int(best_row["epoch"]),
            "best_fitness_csv": best_row.get(fitness_key),
            "best_map50_validation": best_row.get("metrics/mAP50(B)"),
            "best_map50_95_validation": best_row.get(map_key),
            "test_map50_yolov10": yolo_metrics.get("metrics/mAP50(B)"),
            "test_map50_95_yolov10": yolo_metrics.get(
                "metrics/mAP50-95(B)"
            ),
            "test_map50_coco_recomputed": recomputed.get("map50"),
            "test_map50_95_coco_recomputed": recomputed.get("map50_95"),
        },
        "checks": checks,
        "artifacts": [file_evidence(path) for path in important_files],
    }
    AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
