"""Audit the completed P2-001 experiment against the frozen baseline."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
P2_ROOT = PROJECT_ROOT / "03_实验" / "P2-001"
TRAIN_DIR = P2_ROOT / "train"
TEST_DIR = P2_ROOT / "test_eval"
VAL_DIR = P2_ROOT / "val_eval"
LOG_ROOT = PROJECT_ROOT / "03_实验" / "logs" / "P2-001"
EVIDENCE_ROOT = PROJECT_ROOT / "01_立项" / "阶段4_证据"
SUMMARY_PATH = PROJECT_ROOT / "01_立项" / "阶段总结" / "阶段4_总结.md"
AUDIT_PATH = EVIDENCE_ROOT / "阶段4审计.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def count_labels(label_dir: Path) -> tuple[int, int]:
    files = sorted(label_dir.glob("*.txt"))
    lines = sum(
        sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        for path in files
    )
    return len(files), lines


def main() -> int:
    training = load_json(EVIDENCE_ROOT / "stage4_training_summary.json")
    evaluation = load_json(EVIDENCE_ROOT / "stage4_test_evaluation.json")
    comparison = load_json(EVIDENCE_ROOT / "stage4_comparison.json")
    run_status = load_json(LOG_ROOT / "run_status.json")
    architecture = load_json(EVIDENCE_ROOT / "P2架构检查.json")
    smoke = load_json(EVIDENCE_ROOT / "P2_smoke_summary.json")
    val_manifest = load_json(VAL_DIR / "stage4_val_manifest.json")

    with (TRAIN_DIR / "results.csv").open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header = [value.strip() for value in rows[0]]
    train_rows = [dict(zip(header, row)) for row in rows[1:] if row]
    best_epoch = max(
        train_rows,
        key=lambda row: float(row["metrics/mAP50-95(B)"])
        + 0.1 * float(row["metrics/mAP50(B)"]),
    )
    prediction_files, prediction_lines = count_labels(TEST_DIR / "labels")
    per_class = pd.read_csv(TEST_DIR / "per_class_metrics.csv")
    test_dirs = sorted(P2_ROOT.glob("test_eval*"))
    comparison_metrics = comparison["metric_comparison"]
    yolo_metrics = evaluation["metrics_yolov10"]
    coco_metrics = evaluation["recomputed_metrics"]

    required_artifacts = [
        TRAIN_DIR / "results.csv",
        TRAIN_DIR / "weights" / "best.pt",
        TRAIN_DIR / "weights" / "last.pt",
        TEST_DIR / "metrics_yolov10.json",
        TEST_DIR / "per_class_metrics.csv",
        TEST_DIR / "predictions_category0.json",
        TEST_DIR / "ground_truth_coco.json",
        TEST_DIR / "confusion_matrix.png",
        TRAIN_DIR / "results.png",
        VAL_DIR / "stage4_val_manifest.json",
        EVIDENCE_ROOT / "stage4_training_summary.json",
        EVIDENCE_ROOT / "stage4_test_evaluation.json",
        EVIDENCE_ROOT / "stage4_comparison.json",
        EVIDENCE_ROOT / "阶段4结果分析.md",
        SUMMARY_PATH,
    ]

    checks = [
        check(
            "training_completed",
            training["status"] == "passed"
            and training["completed_epochs"] == 40
            and len(train_rows) == 40
            and training["losses_finite"],
            {
                "status": training["status"],
                "completed_epochs": training["completed_epochs"],
                "results_rows": len(train_rows),
                "losses_finite": training["losses_finite"],
            },
        ),
        check(
            "best_validation_epoch",
            int(best_epoch["epoch"]) == 38
            and float(best_epoch["metrics/mAP50-95(B)"]) == 0.38725,
            {
                "epoch": int(best_epoch["epoch"]),
                "mAP50": float(best_epoch["metrics/mAP50(B)"]),
                "mAP50-95": float(best_epoch["metrics/mAP50-95(B)"]),
            },
        ),
        check(
            "p2_detection_scales",
            training["detect_strides"] == [4.0, 8.0, 16.0, 32.0]
            and architecture["status"] == "passed"
            and smoke["status"] == "passed",
            {
                "training_strides": training["detect_strides"],
                "architecture_status": architecture["status"],
                "smoke_status": smoke["status"],
            },
        ),
        check(
            "test_evaluated_exactly_once",
            len(test_dirs) == 1
            and evaluation["status"] == "passed"
            and evaluation["test_inference_run_count"] == 1
            and run_status["status"] == "passed",
            {
                "test_eval_directories": [str(path) for path in test_dirs],
                "evaluation_status": evaluation["status"],
                "run_status": run_status["status"],
            },
        ),
        check(
            "prediction_export_complete",
            prediction_files == 3071 and prediction_lines == 307100,
            {"files": prediction_files, "lines": prediction_lines},
        ),
        check(
            "per_class_metrics_complete",
            len(per_class) == 45 and per_class["class_id"].nunique() == 45,
            {"rows": len(per_class)},
        ),
        check(
            "yolo_coco_metrics_consistent",
            abs(yolo_metrics["metrics/mAP50(B)"] - coco_metrics["map50"]) < 0.01
            and abs(
                yolo_metrics["metrics/mAP50-95(B)"] - coco_metrics["map50_95"]
            )
            < 0.01,
            {
                "yolo_mAP50": yolo_metrics["metrics/mAP50(B)"],
                "coco_mAP50": coco_metrics["map50"],
                "yolo_mAP50_95": yolo_metrics["metrics/mAP50-95(B)"],
                "coco_mAP50_95": coco_metrics["map50_95"],
            },
        ),
        check(
            "small_object_mechanism_direction",
            comparison_metrics["delta_val_map_small"] > 0
            and comparison_metrics["delta_test_map_small"] > 0,
            {
                "val_small_delta": comparison_metrics["delta_val_map_small"],
                "test_small_delta": comparison_metrics["delta_test_map_small"],
            },
        ),
        check(
            "overall_tradeoff_recorded",
            comparison_metrics["delta_val_map50_95"] < 0
            and comparison_metrics["delta_test_map50_95"] < 0,
            {
                "val_mAP50_95_delta": comparison_metrics["delta_val_map50_95"],
                "test_mAP50_95_delta": comparison_metrics["delta_test_map50_95"],
            },
        ),
        check(
            "validation_diagnostic_ran_on_val_only",
            val_manifest["status"] == "passed"
            and val_manifest["split"] == "val"
            and val_manifest["test_inference_run"] is False,
            {
                "status": val_manifest["status"],
                "split": val_manifest["split"],
                "test_inference_run": val_manifest["test_inference_run"],
            },
        ),
        check(
            "required_artifacts_exist",
            all(path.exists() for path in required_artifacts),
            {
                "missing": [str(path) for path in required_artifacts if not path.exists()]
            },
        ),
        check(
            "negative_result_decision_recorded",
            "作为最终主模型" in SUMMARY_PATH.read_text(encoding="utf-8")
            and "BASELINE-001" in SUMMARY_PATH.read_text(encoding="utf-8"),
            {"summary": str(SUMMARY_PATH)},
        ),
    ]

    artifact_paths = [
        SCRIPT_DIR / "stage4_runner.py",
        SCRIPT_DIR / "analyze_stage4.py",
        TRAIN_DIR / "results.csv",
        TRAIN_DIR / "weights" / "best.pt",
        TEST_DIR / "metrics_yolov10.json",
        TEST_DIR / "per_class_metrics.csv",
        TEST_DIR / "predictions_category0.json",
        EVIDENCE_ROOT / "stage4_training_summary.json",
        EVIDENCE_ROOT / "stage4_test_evaluation.json",
        EVIDENCE_ROOT / "stage4_comparison.json",
        EVIDENCE_ROOT / "阶段4结果分析.md",
        SUMMARY_PATH,
    ]
    artifacts = [
        {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
            "sha256": sha256(path) if path.exists() else None,
        }
        for path in artifact_paths
    ]
    status = "passed" if all(item["passed"] for item in checks) else "failed"
    payload = {
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "stage": 4,
        "experiment_id": "P2-001",
        "status": status,
        "decision": "small-object mechanism directionally supported, but model not selected",
        "checks": checks,
        "artifacts": artifacts,
    }
    AUDIT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "audit": str(AUDIT_PATH)}, ensure_ascii=False))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
