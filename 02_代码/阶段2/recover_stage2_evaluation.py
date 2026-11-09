"""Recover Stage 2 evaluation metrics without repeating test inference.

The first and only test-image forward pass completed and saved 3,071 raw
prediction files. The runner then failed while converting YOLOv10's class-name
dictionary to CSV because the dictionary uses integer keys. This script:

1. parses the already-written YOLOv10 per-class summary from the process log;
2. converts the saved YOLO txt predictions to COCO format;
3. independently recomputes overall COCO bbox metrics;
4. writes the recovered evaluation summary and per-class CSV.

It does not load an image or run model inference.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
EXP_ROOT = PROJECT_ROOT / "03_实验" / "baseline_yolov10n"
TEST_EVAL_DIR = EXP_ROOT / "test_eval"
EVIDENCE_ROOT = PROJECT_ROOT / "01_立项" / "阶段2_证据"
POSTPROCESS_LOG = (
    PROJECT_ROOT / "03_实验" / "logs" / "BASELINE-001" / "postprocess.log"
)
DATA_YAML = (
    PROJECT_ROOT
    / "03_实验"
    / "datasets"
    / "TT100K"
    / "yolo"
    / "tt100k.yaml"
)
FAILED_SUMMARY = EVIDENCE_ROOT / "stage2_test_evaluation.initial_failure.json"
RECOVERED_SUMMARY = EVIDENCE_ROOT / "stage2_test_evaluation.recovered.json"
YOLO_METRICS = TEST_EVAL_DIR / "metrics_yolov10.json"
PER_CLASS_CSV = TEST_EVAL_DIR / "per_class_metrics.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_runner():
    runner_path = SCRIPT_DIR / "stage2_runner.py"
    spec = importlib.util.spec_from_file_location(
        "stage2_runner_for_recovery", runner_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {runner_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_yolov10_table(log_text: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    overall_match = re.findall(
        r"^\s+all\s+(\d+)\s+(\d+)\s+"
        r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$",
        log_text,
        re.MULTILINE,
    )
    if not overall_match:
        raise RuntimeError("Cannot find YOLOv10 overall test metrics in log")
    images, instances, precision, recall, map50, map5095 = overall_match[-1]

    speed_match = re.findall(
        r"Speed: ([0-9.]+)ms preprocess, ([0-9.]+)ms inference, "
        r"([0-9.]+)ms loss, ([0-9.]+)ms postprocess per image",
        log_text,
    )
    if not speed_match:
        raise RuntimeError("Cannot find YOLOv10 speed line in log")
    preprocess, inference, loss, postprocess = speed_match[-1]

    overall = {
        "metrics/precision(B)": float(precision),
        "metrics/recall(B)": float(recall),
        "metrics/mAP50(B)": float(map50),
        "metrics/mAP50-95(B)": float(map5095),
        "fitness": 0.1 * float(map50) + 0.9 * float(map5095),
        "preprocess_ms_per_image": float(preprocess),
        "inference_ms_per_image": float(inference),
        "loss_ms_per_image": float(loss),
        "postprocess_ms_per_image": float(postprocess),
        "images": int(images),
        "instances": int(instances),
        "source": "recovered from existing test_eval/postprocess console table",
    }

    row_pattern = re.compile(
        r"^\s+(\S+)\s+(\d+)\s+(\d+)\s+"
        r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$"
    )
    rows: list[dict[str, Any]] = []
    for line in log_text.splitlines():
        match = row_pattern.match(line)
        if not match or match.group(1) == "all":
            continue
        name, class_images, class_instances, p, r, ap50, ap5095 = match.groups()
        if int(class_images) != 3071:
            continue
        rows.append(
            {
                "class_name": name,
                "images": int(class_images),
                "instances": int(class_instances),
                "precision": float(p),
                "recall": float(r),
                "map50": float(ap50),
                "map50_95": float(ap5095),
            }
        )
    deduplicated = {row["class_name"]: row for row in rows}
    ordered_names = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))["names"]
    per_class = [
        {**deduplicated[name], "class_id": class_id}
        for class_id, name in enumerate(ordered_names)
        if name in deduplicated
    ]
    if len(per_class) != 45:
        raise RuntimeError(f"Expected 45 per-class rows, found {len(per_class)}")
    return overall, per_class


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "class_id",
        "class_name",
        "precision",
        "recall",
        "map50",
        "map50_95",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    failed_summary = json.loads(FAILED_SUMMARY.read_text(encoding="utf-8"))
    if failed_summary.get("status") != "failed":
        raise RuntimeError("Expected the original evaluation summary to be failed")
    if "KeyError: '0'" not in str(failed_summary.get("error")):
        raise RuntimeError("Unexpected original evaluation failure")

    prediction_dir = TEST_EVAL_DIR / "labels"
    prediction_files = sorted(prediction_dir.glob("*.txt"))
    if len(prediction_files) != 3071:
        raise RuntimeError(
            f"Expected 3,071 saved prediction files, found {len(prediction_files)}"
        )

    log_text = POSTPROCESS_LOG.read_text(encoding="utf-8", errors="replace")
    yolo_metrics, per_class = parse_yolov10_table(log_text)
    YOLO_METRICS.write_text(
        json.dumps(yolo_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(PER_CLASS_CSV, per_class)

    runner = load_runner()
    recomputed = runner.recompute_metrics(
        DATA_YAML,
        "test",
        TEST_EVAL_DIR,
        prediction_dir,
    )

    checkpoint = EXP_ROOT / "train" / "weights" / "best.pt"
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    checkpoint_model = checkpoint_data.get("ema") or checkpoint_data["model"]
    parameter_count = sum(
        parameter.numel() for parameter in checkpoint_model.parameters()
    )
    failed_finished = datetime.fromisoformat(failed_summary["finished_at"])
    summary = {
        "experiment_id": "BASE_YOLOV10N_TT100K45_S2026",
        "started_at": failed_summary["started_at"],
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "passed",
        "checkpoint": str(checkpoint),
        "data_yaml": str(DATA_YAML),
        "split": "test",
        "selection_rule": "best.pt selected by validation fitness; test evaluated once",
        "model": {
            "parameters": int(parameter_count),
            "source": "best.pt model after optimizer stripping",
        },
        "metrics_yolov10": yolo_metrics,
        "per_class_metrics": str(PER_CLASS_CSV),
        "raw_prediction_labels": str(prediction_dir),
        "raw_prediction_file_count": len(prediction_files),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "recomputed_metrics": recomputed,
        "raw_predictions": str(TEST_EVAL_DIR / "predictions_category0.json"),
        "elapsed_seconds": round(
            float(failed_summary["elapsed_seconds"])
            + (datetime.now().astimezone() - failed_finished).total_seconds(),
            3,
        ),
        "recovery": {
            "required": True,
            "reason": (
                "唯一一次测试前向、原始预测和图表已正常保存；评估汇总在把 "
                "YOLOv10 整数键类别字典转换为每类 CSV 时触发 KeyError。"
            ),
            "test_inference_rerun": False,
            "source_failed_summary": str(FAILED_SUMMARY),
            "source_failed_traceback": failed_summary.get("traceback"),
            "source_postprocess_log": str(POSTPROCESS_LOG),
            "source_postprocess_log_sha256": sha256(POSTPROCESS_LOG),
            "source_prediction_file_count": len(prediction_files),
        },
    }
    RECOVERED_SUMMARY.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Recovered evaluation summary: {RECOVERED_SUMMARY}")
    print(f"Status: {summary['status']}")
    print(json.dumps(recomputed, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
