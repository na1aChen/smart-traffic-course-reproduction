"""Audit Stage 5 multi-seed and training-budget evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
ROBUST_ROOT = PROJECT_ROOT / "03_实验" / "robustness"
EVIDENCE_ROOT = PROJECT_ROOT / "01_立项" / "阶段5_证据"
LOG_ROOT = PROJECT_ROOT / "03_实验" / "logs" / "STAGE5"
SUMMARY_PATH = PROJECT_ROOT / "01_立项" / "阶段总结" / "阶段5_总结.md"
AUDIT_PATH = EVIDENCE_ROOT / "阶段5审计.json"

EXPERIMENTS = {
    "BASELINE-002-S2027": 40,
    "P2-002-S2027": 40,
    "BASELINE-BUDGET48-S2026": 48,
}


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
    queue = load_json(LOG_ROOT / "queue_status.json")
    comparison = load_json(EVIDENCE_ROOT / "stage5_comparison.json")
    metrics_frame = pd.read_csv(EVIDENCE_ROOT / "stage5_experiment_metrics.csv")
    paired = pd.read_csv(EVIDENCE_ROOT / "stage5_paired_deltas.csv")

    experiment_checks: dict[str, Any] = {}
    required_paths: list[Path] = [
        EVIDENCE_ROOT / "阶段5预注册.md",
        EVIDENCE_ROOT / "阶段5结果分析.md",
        EVIDENCE_ROOT / "阶段5断电保护说明.md",
        EVIDENCE_ROOT / "手动暂停记录.json",
        EVIDENCE_ROOT / "stage5_queue_status.json",
        EVIDENCE_ROOT / "stage5_comparison.json",
        EVIDENCE_ROOT / "stage5_experiment_metrics.csv",
        EVIDENCE_ROOT / "stage5_paired_deltas.csv",
        EVIDENCE_ROOT / "stage5_seed_comparison.png",
        SUMMARY_PATH,
    ]

    for experiment_id, expected_epochs in EXPERIMENTS.items():
        train_summary_path = EVIDENCE_ROOT / "training" / f"{experiment_id}.json"
        eval_summary_path = EVIDENCE_ROOT / "evaluation" / f"{experiment_id}.json"
        val_summary_path = EVIDENCE_ROOT / "validation" / f"{experiment_id}.json"
        train_summary = load_json(train_summary_path)
        eval_summary = load_json(eval_summary_path)
        val_summary = load_json(val_summary_path)
        run_root = ROBUST_ROOT / experiment_id
        test_dir = run_root / "test_eval"
        prediction_files, prediction_lines = count_labels(test_dir / "labels")
        per_class = pd.read_csv(test_dir / "per_class_metrics.csv")
        test_roots = sorted(run_root.glob("test_eval*"))
        yolo = eval_summary["metrics_yolov10"]
        coco = eval_summary["recomputed_metrics"]
        experiment_checks[experiment_id] = {
            "training": check(
                f"{experiment_id}_training",
                train_summary["status"] == "passed"
                and train_summary["completed_epochs"] == expected_epochs,
                {
                    "status": train_summary["status"],
                    "completed_epochs": train_summary["completed_epochs"],
                    "best_epoch": train_summary.get("best_epoch"),
                },
            ),
            "test_once": check(
                f"{experiment_id}_test_once",
                len(test_roots) == 1
                and eval_summary["status"] == "passed"
                and eval_summary["test_inference_run_count"] == 1,
                {"test_directories": [str(path) for path in test_roots]},
            ),
            "predictions": check(
                f"{experiment_id}_predictions",
                prediction_files == 3071 and prediction_lines == 307100,
                {"files": prediction_files, "lines": prediction_lines},
            ),
            "per_class": check(
                f"{experiment_id}_per_class",
                len(per_class) == 45 and per_class["class_id"].nunique() == 45,
                {"rows": len(per_class)},
            ),
            "metric_consistency": check(
                f"{experiment_id}_metric_consistency",
                abs(yolo["metrics/mAP50(B)"] - coco["map50"]) < 0.01
                and abs(yolo["metrics/mAP50-95(B)"] - coco["map50_95"]) < 0.01,
                {
                    "yolo_map50": yolo["metrics/mAP50(B)"],
                    "coco_map50": coco["map50"],
                    "yolo_map50_95": yolo["metrics/mAP50-95(B)"],
                    "coco_map50_95": coco["map50_95"],
                },
            ),
            "validation_diagnostic": check(
                f"{experiment_id}_validation",
                val_summary["status"] == "passed"
                and val_summary["split"] == "val"
                and val_summary["test_inference_run"] is False,
                {
                    "split": val_summary["split"],
                    "test_inference_run": val_summary["test_inference_run"],
                },
            ),
        }
        required_paths.extend(
            [
                train_summary_path,
                eval_summary_path,
                val_summary_path,
                test_dir / "metrics_yolov10.json",
                test_dir / "per_class_metrics.csv",
                test_dir / "predictions_category0.json",
                test_dir / "confusion_matrix.png",
                run_root / "train" / "results.csv",
                run_root / "train" / "weights" / "best.pt",
            ]
        )

    baseline_rows = metrics_frame[metrics_frame["group"] == "baseline"]
    p2_rows = metrics_frame[metrics_frame["group"] == "p2"]
    budget_rows = metrics_frame[metrics_frame["group"] == "budget"]
    checks = {
        "queue_complete": check(
            "queue_complete",
            queue["status"] == "passed"
            and queue["completed_count"] == 3
            and queue["current_experiment"] is None,
            queue,
        ),
        "all_new_experiments": {
            experiment_id: item
            for experiment_id, values in experiment_checks.items()
            for item in values.values()
        },
        "two_seeds_per_family": check(
            "two_seeds_per_family",
            set(baseline_rows["seed"]) == {2026, 2027}
            and set(p2_rows["seed"]) == {2026, 2027},
            {
                "baseline_seeds": sorted(baseline_rows["seed"].tolist()),
                "p2_seeds": sorted(p2_rows["seed"].tolist()),
            },
        ),
        "small_gain_reproduced": check(
            "small_gain_reproduced",
            bool((paired["delta_test_map_small"] > 0).all()),
            paired[
                ["seed", "delta_test_map50", "delta_test_map50_95", "delta_test_map_small"]
            ].to_dict(orient="records"),
        ),
        "budget_control_present": check(
            "budget_control_present",
            len(budget_rows) == 1
            and int(budget_rows.iloc[0]["epochs"]) == 48
            and float(budget_rows.iloc[0]["test_map_small"])
            < comparison["groups"]["p2"]["test_map_small"]["mean"],
            budget_rows.iloc[0].to_dict(),
        ),
        "final_model_unchanged": check(
            "final_model_unchanged",
            comparison["interpretation"]["final_model"] == "BASELINE-001"
            and "最终主模型保持 `BASELINE-001`"
            in SUMMARY_PATH.read_text(encoding="utf-8"),
            {"final_model": comparison["interpretation"]["final_model"]},
        ),
        "required_artifacts": check(
            "required_artifacts",
            all(path.exists() for path in required_paths),
            {"missing": [str(path) for path in required_paths if not path.exists()]},
        ),
    }
    flattened_checks = []
    for key, value in checks.items():
        if key == "all_new_experiments":
            flattened_checks.extend(value.values())
        else:
            flattened_checks.append(value)
    status = "passed" if all(item["passed"] for item in flattened_checks) else "failed"

    artifacts = []
    for path in required_paths:
        artifacts.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "sha256": sha256(path) if path.exists() else None,
            }
        )
    payload = {
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "stage": 5,
        "status": status,
        "decision": {
            "small_gain_reproduced": True,
            "final_model": "BASELINE-001",
            "p2_selected": False,
        },
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
