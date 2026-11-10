"""Reproducible multi-seed and training-budget analysis for Stage 5."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
STAGE2_CODE = PROJECT_ROOT / "02_代码" / "阶段2"
STAGE3_CODE = PROJECT_ROOT / "02_代码" / "阶段3"
SOURCE_ROOT = PROJECT_ROOT / "02_代码" / "源码" / "yolov10"
DATA_DIR = PROJECT_ROOT / "03_实验" / "datasets" / "TT100K" / "yolo"
DATA_YAML = DATA_DIR / "tt100k.yaml"
BASELINE_ROOT = PROJECT_ROOT / "03_实验" / "baseline_yolov10n"
P2_ROOT = PROJECT_ROOT / "03_实验" / "P2-001"
ROBUST_ROOT = PROJECT_ROOT / "03_实验" / "robustness"
STAGE2_EVIDENCE = PROJECT_ROOT / "01_立项" / "阶段2_证据"
STAGE3_EVIDENCE = PROJECT_ROOT / "01_立项" / "阶段3_证据"
STAGE4_EVIDENCE = PROJECT_ROOT / "01_立项" / "阶段4_证据"
STAGE5_EVIDENCE = PROJECT_ROOT / "01_立项" / "阶段5_证据"
VAL_DIAG_ROOT = STAGE5_EVIDENCE / "validation"

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "BASELINE-001": {
        "group": "baseline",
        "seed": 2026,
        "epochs": 40,
        "run_root": BASELINE_ROOT,
        "training_summary": STAGE2_EVIDENCE / "stage2_training_summary.json",
        "test_summary": STAGE2_EVIDENCE / "stage2_test_evaluation.json",
        "test_labels": BASELINE_ROOT / "test_eval" / "labels",
        "val_summary": STAGE3_EVIDENCE / "stage3_baseline_val_coco_metrics.json",
        "checkpoint": BASELINE_ROOT / "train" / "weights" / "best.pt",
    },
    "P2-001": {
        "group": "p2",
        "seed": 2026,
        "epochs": 40,
        "run_root": P2_ROOT,
        "training_summary": STAGE4_EVIDENCE / "stage4_training_summary.json",
        "test_summary": STAGE4_EVIDENCE / "stage4_test_evaluation.json",
        "test_labels": P2_ROOT / "test_eval" / "labels",
        "val_summary": (
            PROJECT_ROOT
            / "03_实验"
            / "P2-001"
            / "val_eval"
            / "stage4_val_manifest.json"
        ),
        "checkpoint": P2_ROOT / "train" / "weights" / "best.pt",
    },
    "BASELINE-002-S2027": {
        "group": "baseline",
        "seed": 2027,
        "epochs": 40,
        "run_root": ROBUST_ROOT / "BASELINE-002-S2027",
        "training_summary": (
            STAGE5_EVIDENCE / "training" / "BASELINE-002-S2027.json"
        ),
        "test_summary": (
            STAGE5_EVIDENCE / "evaluation" / "BASELINE-002-S2027.json"
        ),
        "test_labels": ROBUST_ROOT / "BASELINE-002-S2027" / "test_eval" / "labels",
        "val_summary": VAL_DIAG_ROOT / "BASELINE-002-S2027.json",
        "checkpoint": (
            ROBUST_ROOT / "BASELINE-002-S2027" / "train" / "weights" / "best.pt"
        ),
    },
    "P2-002-S2027": {
        "group": "p2",
        "seed": 2027,
        "epochs": 40,
        "run_root": ROBUST_ROOT / "P2-002-S2027",
        "training_summary": STAGE5_EVIDENCE / "training" / "P2-002-S2027.json",
        "test_summary": STAGE5_EVIDENCE / "evaluation" / "P2-002-S2027.json",
        "test_labels": ROBUST_ROOT / "P2-002-S2027" / "test_eval" / "labels",
        "val_summary": VAL_DIAG_ROOT / "P2-002-S2027.json",
        "checkpoint": ROBUST_ROOT / "P2-002-S2027" / "train" / "weights" / "best.pt",
    },
    "BASELINE-BUDGET48-S2026": {
        "group": "budget",
        "seed": 2026,
        "epochs": 48,
        "run_root": ROBUST_ROOT / "BASELINE-BUDGET48-S2026",
        "training_summary": (
            STAGE5_EVIDENCE / "training" / "BASELINE-BUDGET48-S2026.json"
        ),
        "test_summary": (
            STAGE5_EVIDENCE / "evaluation" / "BASELINE-BUDGET48-S2026.json"
        ),
        "test_labels": (
            ROBUST_ROOT / "BASELINE-BUDGET48-S2026" / "test_eval" / "labels"
        ),
        "val_summary": VAL_DIAG_ROOT / "BASELINE-BUDGET48-S2026.json",
        "checkpoint": (
            ROBUST_ROOT
            / "BASELINE-BUDGET48-S2026"
            / "train"
            / "weights"
            / "best.pt"
        ),
    },
}

sys.path.insert(0, str(STAGE2_CODE))
sys.path.insert(0, str(STAGE3_CODE))
sys.path.insert(0, str(SOURCE_ROOT))

from stage2_runner import (  # noqa: E402
    install_unicode_path_save_patch,
    recompute_metrics,
    set_reproducibility,
    tee_console,
)
from diagnose_stage3 import (  # noqa: E402
    OPERATIONAL_CONF,
    add_names,
    add_size_buckets,
    build_confusion_table,
    collect_events,
    count_summary,
    load_names,
    load_split,
)
from ultralytics import YOLOv10  # noqa: E402


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def prepare_validation() -> int:
    install_unicode_path_save_patch()
    VAL_DIAG_ROOT.mkdir(parents=True, exist_ok=True)
    for experiment_id, definition in EXPERIMENTS.items():
        summary_path = Path(definition["val_summary"])
        if summary_path.is_file():
            print(f"Validation diagnostic exists: {experiment_id}")
            continue
        checkpoint = Path(definition["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        eval_dir = Path(definition["run_root"]) / "val_eval_diag"
        if eval_dir.exists():
            raise FileExistsError(f"Validation diagnostic directory exists: {eval_dir}")
        eval_dir.mkdir(parents=True, exist_ok=False)
        set_reproducibility(int(definition["seed"]))
        model = YOLOv10(str(checkpoint))
        with tee_console(eval_dir / "console.log"):
            metrics = model.val(
                data=str(DATA_YAML),
                split="val",
                imgsz=640,
                batch=8,
                device="0",
                workers=0,
                project=str(Path(definition["run_root"])),
                name="val_eval_diag",
                exist_ok=True,
                save_json=False,
                save_txt=True,
                save_conf=True,
                plots=True,
                conf=0.001,
                iou=0.7,
                max_det=100,
                seed=int(definition["seed"]),
                deterministic=True,
                verbose=True,
            )
        coco = recompute_metrics(DATA_YAML, "val", eval_dir, eval_dir / "labels")
        payload = {
            "created_at": datetime.now(timezone.utc).astimezone().isoformat(
                timespec="seconds"
            ),
            "experiment_id": experiment_id,
            "status": "passed",
            "split": "val",
            "checkpoint": str(checkpoint),
            "test_inference_run": False,
            "metrics_yolov10": {
                str(key): float(value) for key, value in metrics.results_dict.items()
            },
            "recomputed_metrics": coco,
        }
        write_json(summary_path, payload)
        print(f"Validation diagnostic: {summary_path}")
    return 0


def load_metric_source(
    experiment_id: str, definition: dict[str, Any]
) -> dict[str, Any]:
    if experiment_id == "BASELINE-001":
        test = json.loads(
            Path(definition["test_summary"]).read_text(encoding="utf-8")
        )["recomputed_metrics"]
        val_payload = json.loads(
            Path(definition["val_summary"]).read_text(encoding="utf-8")
        )
        val = val_payload
    elif experiment_id == "P2-001":
        test = json.loads(
            Path(definition["test_summary"]).read_text(encoding="utf-8")
        )["recomputed_metrics"]
        val = json.loads(
            Path(definition["val_summary"]).read_text(encoding="utf-8")
        )["recomputed_metrics"]
    else:
        test = json.loads(
            Path(definition["test_summary"]).read_text(encoding="utf-8")
        )["recomputed_metrics"]
        val = json.loads(
            Path(definition["val_summary"]).read_text(encoding="utf-8")
        )["recomputed_metrics"]
    train = json.loads(
        Path(definition["training_summary"]).read_text(encoding="utf-8")
    )
    if not train.get("best_validation") or train.get("best_epoch") is None:
        results_path = Path(definition["run_root"]) / "train" / "results.csv"
        with results_path.open(encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = [value.strip() for value in next(reader)]
            rows = [
                dict(zip(header, [value.strip() for value in row]))
                for row in reader
                if row
            ]
        best = max(
            rows,
            key=lambda row: float(row["metrics/mAP50-95(B)"])
            + 0.1 * float(row["metrics/mAP50(B)"]),
        )
        train["best_epoch"] = int(best["epoch"])
        train["best_validation"] = {
            "precision": float(best["metrics/precision(B)"]),
            "recall": float(best["metrics/recall(B)"]),
            "map50": float(best["metrics/mAP50(B)"]),
            "map50_95": float(best["metrics/mAP50-95(B)"]),
        }
    return {
        "test": test,
        "val": val,
        "training": train,
    }


def analyze_operational(experiment_id: str, definition: dict[str, Any]) -> dict[str, Any]:
    names = load_names()
    gt, pred, _ = load_split(
        experiment_id,
        DATA_DIR / "labels" / "test",
        Path(definition["test_labels"]),
        DATA_DIR / "images" / "test",
    )
    gt_events, pred_events = collect_events(
        experiment_id, gt, pred, OPERATIONAL_CONF
    )
    gt_events = add_names(add_size_buckets(gt_events), names)
    pred_events = add_names(pred_events, names)
    summary = count_summary(gt_events, pred_events)
    confusion = build_confusion_table(gt_events)
    confusion.head(20).to_csv(
        STAGE5_EVIDENCE / f"operational_{experiment_id}_top_confusions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return summary


def mean_std(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) >= 2 else None,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def analyze() -> int:
    sources = {
        experiment_id: load_metric_source(experiment_id, definition)
        for experiment_id, definition in EXPERIMENTS.items()
    }
    operational = {
        experiment_id: analyze_operational(experiment_id, definition)
        for experiment_id, definition in EXPERIMENTS.items()
    }

    rows: list[dict[str, Any]] = []
    for experiment_id, definition in EXPERIMENTS.items():
        source = sources[experiment_id]
        test = source["test"]
        val = source["val"]
        best = source["training"].get("best_validation") or {}
        rows.append(
            {
                "experiment_id": experiment_id,
                "group": definition["group"],
                "seed": definition["seed"],
                "epochs": definition["epochs"],
                "best_epoch": source["training"].get("best_epoch"),
                "val_map50": best.get("map50"),
                "val_map50_95": best.get("map50_95"),
                "val_map_small": val.get("map_small"),
                "test_map50": test.get("map50"),
                "test_map50_95": test.get("map50_95"),
                "test_map_small": test.get("map_small"),
                "test_map_medium": test.get("map_medium"),
                "test_map_large": test.get("map_large"),
                "test_mar100": test.get("mar100"),
                "tp_at_0.25": operational[experiment_id]["tp"],
                "confusion_at_0.25": operational[experiment_id]["confusion"],
                "missed_at_0.25": operational[experiment_id]["missed"],
                "background_fp_at_0.25": operational[experiment_id]["background_fp"],
                "duplicate_at_0.25": operational[experiment_id]["duplicate"],
            }
        )
    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(
        STAGE5_EVIDENCE / "stage5_experiment_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    baseline_frame = metrics_frame[metrics_frame["group"] == "baseline"]
    p2_frame = metrics_frame[metrics_frame["group"] == "p2"]
    budget_frame = metrics_frame[metrics_frame["group"] == "budget"]
    metric_columns = [
        "val_map50_95",
        "val_map_small",
        "test_map50",
        "test_map50_95",
        "test_map_small",
    ]
    groups: dict[str, Any] = {}
    for group_name, frame in (("baseline", baseline_frame), ("p2", p2_frame)):
        groups[group_name] = {
            column: mean_std(frame[column].astype(float).tolist())
            for column in metric_columns
        }

    paired = baseline_frame[["seed", "test_map50", "test_map50_95", "test_map_small"]].merge(
        p2_frame[["seed", "test_map50", "test_map50_95", "test_map_small"]],
        on="seed",
        suffixes=("_baseline", "_p2"),
    )
    paired["delta_test_map50"] = paired["test_map50_p2"] - paired["test_map50_baseline"]
    paired["delta_test_map50_95"] = (
        paired["test_map50_95_p2"] - paired["test_map50_95_baseline"]
    )
    paired["delta_test_map_small"] = (
        paired["test_map_small_p2"] - paired["test_map_small_baseline"]
    )
    paired.to_csv(
        STAGE5_EVIDENCE / "stage5_paired_deltas.csv",
        index=False,
        encoding="utf-8-sig",
    )

    budget = budget_frame.iloc[0].to_dict()
    comparison = {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"
        ),
        "purpose": "Stage 5 multi-seed and training-budget robustness audit",
        "operational_confidence": OPERATIONAL_CONF,
        "experiments": rows,
        "groups": groups,
        "paired_deltas": paired.to_dict(orient="records"),
        "budget_control": budget,
        "interpretation": {
            "small_gain_reproduced_on_test": bool(
                (paired["delta_test_map_small"] > 0).all()
            ),
            "p2_overall_mean_delta_map50": (
                groups["p2"]["test_map50"]["mean"]
                - groups["baseline"]["test_map50"]["mean"]
            ),
            "p2_overall_mean_delta_map50_95": (
                groups["p2"]["test_map50_95"]["mean"]
                - groups["baseline"]["test_map50_95"]["mean"]
            ),
            "p2_small_mean_delta_map50_95": (
                groups["p2"]["test_map_small"]["mean"]
                - groups["baseline"]["test_map_small"]["mean"]
            ),
            "budget48_map_small": budget["test_map_small"],
            "budget48_map50_95": budget["test_map50_95"],
            "final_model": "BASELINE-001",
        },
    }
    write_json(STAGE5_EVIDENCE / "stage5_comparison.json", comparison)

    figure_path = STAGE5_EVIDENCE / "stage5_seed_comparison.png"
    labels = ["mAP50", "mAP50-95", "mAP small"]
    keys = ["test_map50", "test_map50_95", "test_map_small"]
    positions = list(range(len(keys)))
    fig, axis = plt.subplots(figsize=(9, 5.5))
    axis.bar(
        [x - 0.2 for x in positions],
        [groups["baseline"][key]["mean"] for key in keys],
        yerr=[groups["baseline"][key]["std"] or 0 for key in keys],
        width=0.4,
        capsize=4,
        label="Baseline mean ± SD (n=2)",
    )
    axis.bar(
        [x + 0.2 for x in positions],
        [groups["p2"][key]["mean"] for key in keys],
        yerr=[groups["p2"][key]["std"] or 0 for key in keys],
        width=0.4,
        capsize=4,
        label="P2 mean ± SD (n=2)",
    )
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 0.6)
    axis.set_ylabel("COCO metric")
    axis.set_title("Two-seed robustness comparison")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    comparison["figure"] = str(figure_path)
    write_json(STAGE5_EVIDENCE / "stage5_comparison.json", comparison)

    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare-val", "analyze"))
    args = parser.parse_args()
    if args.command == "prepare-val":
        return prepare_validation()
    return analyze()


if __name__ == "__main__":
    raise SystemExit(main())
