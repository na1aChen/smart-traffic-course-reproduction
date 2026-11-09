"""Audit Stage 3 diagnostic evidence without running the model."""

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
ANALYSIS_DIR = PROJECT_ROOT / "03_实验" / "诊断" / "BASELINE-001" / "analysis"
CASE_DIR = ANALYSIS_DIR / "cases"
VAL_MANIFEST = (
    PROJECT_ROOT
    / "03_实验"
    / "诊断"
    / "BASELINE-001"
    / "val_eval"
    / "stage3_val_manifest.json"
)
TEST_EVAL_DIR = PROJECT_ROOT / "03_实验" / "baseline_yolov10n" / "test_eval"
EVIDENCE_DIR = PROJECT_ROOT / "01_立项" / "阶段3_证据"
SUMMARY_PATH = PROJECT_ROOT / "01_立项" / "阶段总结" / "阶段3_总结.md"
AUDIT_PATH = EVIDENCE_DIR / "阶段3审计.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "detail": detail}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def count_prediction_lines(label_dir: Path) -> tuple[int, int]:
    files = sorted(label_dir.glob("*.txt"))
    lines = sum(
        1
        for path in files
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return len(files), lines


def main() -> int:
    summary = load_json(ANALYSIS_DIR / "analysis_summary.json")
    val_manifest = load_json(VAL_MANIFEST)
    per_class = pd.read_csv(ANALYSIS_DIR / "per_class_diagnostics.csv")
    size_table = pd.read_csv(ANALYSIS_DIR / "size_bucket_metrics.csv")
    confusion = pd.read_csv(ANALYSIS_DIR / "confusion_pairs.csv")
    cases = pd.read_csv(ANALYSIS_DIR / "case_manifest.csv")
    prediction_file_count, prediction_line_count = count_prediction_lines(
        TEST_EVAL_DIR / "labels"
    )

    test_per_class = per_class[per_class["split"] == "test"]
    test_size = size_table[size_table["split"] == "test"]
    test_confusion = confusion[confusion["split"] == "test"]
    top_confusion = test_confusion.iloc[0] if len(test_confusion) else None
    case_paths = [Path(value) for value in cases["output_path"].tolist()]
    case_files = sorted(CASE_DIR.glob("case_*.png"))

    val_started = datetime.fromisoformat(val_manifest["created_at"])
    frozen_files = [
        TEST_EVAL_DIR / "predictions_category0.json",
        TEST_EVAL_DIR / "metrics_yolov10.json",
        TEST_EVAL_DIR / "per_class_metrics.csv",
    ]
    latest_frozen_mtime = max(
        datetime.fromtimestamp(path.stat().st_mtime).astimezone()
        for path in frozen_files
    )

    summary_text = SUMMARY_PATH.read_text(encoding="utf-8")
    checks = {
        "frozen_test_consistent": check(
            "frozen_test_consistent",
            bool(summary["frozen_test_check"]["consistent"])
            and summary["frozen_test_check"]["test_inference_rerun"] is False,
            summary["frozen_test_check"],
        ),
        "test_predictions_not_rerun_during_stage3": check(
            "test_predictions_not_rerun_during_stage3",
            latest_frozen_mtime < val_started,
            {
                "latest_stage2_frozen_write": latest_frozen_mtime.isoformat(),
                "stage3_val_start": val_started.isoformat(),
            },
        ),
        "prediction_export_complete": check(
            "prediction_export_complete",
            prediction_file_count == 3071 and prediction_line_count == 307100,
            {
                "files": prediction_file_count,
                "lines": prediction_line_count,
            },
        ),
        "validation_confirmation_ran_on_val_only": check(
            "validation_confirmation_ran_on_val_only",
            val_manifest["status"] == "passed"
            and val_manifest["split"] == "val"
            and "test split was not run" in val_manifest["purpose"],
            val_manifest,
        ),
        "all_45_test_classes_present": check(
            "all_45_test_classes_present",
            len(test_per_class) == 45
            and test_per_class["class_id"].nunique() == 45,
            {"rows": len(test_per_class)},
        ),
        "size_buckets_present": check(
            "size_buckets_present",
            {"small", "medium", "large"}.issubset(
                set(test_size.loc[test_size["bucket_type"] == "coco", "bucket"])
            ),
            test_size.to_dict(orient="records"),
        ),
        "small_target_mechanism_reproduced": check(
            "small_target_mechanism_reproduced",
            summary["mechanism_snapshot"]["test"]["small_tp_recall"]
            < summary["mechanism_snapshot"]["test"]["medium_tp_recall"]
            and summary["mechanism_snapshot"]["val"]["small_tp_recall"]
            < summary["mechanism_snapshot"]["val"]["medium_tp_recall"]
            and summary["mechanism_snapshot"]["test"][
                "within_class_size_control"
            ]["classes_with_medium_recall_above_small"]
            >= 0.8
            * summary["mechanism_snapshot"]["test"]["within_class_size_control"][
                "classes_with_at_least_5_small_and_5_medium"
            ],
            summary["mechanism_snapshot"],
        ),
        "primary_confusion_pair_present": check(
            "primary_confusion_pair_present",
            top_confusion is not None
            and top_confusion["true_class"] == "pl60"
            and top_confusion["predicted_class"] == "pl40"
            and int(top_confusion["count"]) == 52,
            top_confusion.to_dict() if top_confusion is not None else None,
        ),
        "eight_deterministic_cases_present": check(
            "eight_deterministic_cases_present",
            len(cases) == 8
            and len(case_files) == 8
            and all(path.exists() for path in case_paths),
            {
                "manifest_rows": len(cases),
                "case_files": [str(path) for path in case_files],
            },
        ),
        "required_documents_present": check(
            "required_documents_present",
            all(
                path.exists()
                for path in (
                    EVIDENCE_DIR / "阶段3诊断设计.md",
                    EVIDENCE_DIR / "诊断结果.md",
                    EVIDENCE_DIR / "阶段3图表QA.md",
                    SUMMARY_PATH,
                )
            ),
            {
                "design": str(EVIDENCE_DIR / "阶段3诊断设计.md"),
                "diagnosis": str(EVIDENCE_DIR / "诊断结果.md"),
                "visual_qa": str(EVIDENCE_DIR / "阶段3图表QA.md"),
                "summary": str(SUMMARY_PATH),
            },
        ),
        "summary_names_single_factor_candidate": check(
            "summary_names_single_factor_candidate",
            "P2" in summary_text
            and "单一开关" in summary_text
            and "阶段 4" in summary_text,
            {"summary": str(SUMMARY_PATH)},
        ),
        "visual_qa_records_alignment_fix": check(
            "visual_qa_records_alignment_fix",
            "案例框对齐" in (EVIDENCE_DIR / "阶段3图表QA.md").read_text(
                encoding="utf-8"
            )
            and "没有重跑测试推理" in (EVIDENCE_DIR / "阶段3图表QA.md").read_text(
                encoding="utf-8"
            ),
            {"visual_qa": str(EVIDENCE_DIR / "阶段3图表QA.md")},
        ),
    }

    artifact_paths = [
        SCRIPT_DIR / "diagnose_stage3.py",
        ANALYSIS_DIR / "analysis_summary.json",
        ANALYSIS_DIR / "per_class_diagnostics.csv",
        ANALYSIS_DIR / "size_bucket_metrics.csv",
        ANALYSIS_DIR / "confusion_pairs.csv",
        ANALYSIS_DIR / "threshold_sweep.csv",
        ANALYSIS_DIR / "case_manifest.csv",
        ANALYSIS_DIR / "test_gt_events.csv",
        ANALYSIS_DIR / "test_pred_events.csv",
        ANALYSIS_DIR / "figures" / "case_montage.png",
        EVIDENCE_DIR / "阶段3诊断设计.md",
        EVIDENCE_DIR / "诊断结果.md",
        EVIDENCE_DIR / "阶段3图表QA.md",
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
    status = "passed" if all(item["passed"] for item in checks.values()) else "failed"
    payload = {
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "stage": 3,
        "experiment_id": "BASELINE-001",
        "status": status,
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
