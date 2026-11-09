"""Validation diagnostics and baseline comparison for P2-001."""

from __future__ import annotations

import argparse
import json
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
STAGE2_EVIDENCE = PROJECT_ROOT / "01_立项" / "阶段2_证据"
STAGE3_EVIDENCE = PROJECT_ROOT / "01_立项" / "阶段3_证据"
STAGE4_EVIDENCE = PROJECT_ROOT / "01_立项" / "阶段4_证据"
P2_VAL_DIR = P2_ROOT / "val_eval"
BASELINE_VAL_METRICS_DIR = (
    PROJECT_ROOT / "03_实验" / "诊断" / "BASELINE-001" / "val_metric_recompute"
)

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
    build_per_class_table,
    collect_events,
    count_summary,
    load_names,
    load_split,
    summarize_size_buckets,
)
from ultralytics import YOLOv10  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def prepare_val(args: argparse.Namespace) -> int:
    install_unicode_path_save_patch()
    checkpoint = P2_ROOT / "train" / "weights" / "best.pt"
    manifest_path = P2_VAL_DIR / "stage4_val_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") == "passed":
            print(f"P2 validation diagnostic already complete: {manifest_path}")
            return 0
    if P2_VAL_DIR.exists():
        raise FileExistsError(f"P2 validation directory exists: {P2_VAL_DIR}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    P2_VAL_DIR.mkdir(parents=True, exist_ok=False)
    set_reproducibility(args.seed)
    model = YOLOv10(str(checkpoint))
    with tee_console(P2_VAL_DIR / "console.log"):
        metrics = model.val(
            data=str(DATA_YAML),
            split="val",
            imgsz=640,
            batch=8,
            device="0",
            workers=0,
            project=str(P2_ROOT),
            name="val_eval",
            exist_ok=True,
            save_json=False,
            save_txt=True,
            save_conf=True,
            plots=True,
            conf=0.001,
            iou=0.7,
            max_det=100,
            seed=args.seed,
            deterministic=True,
            verbose=True,
        )
    coco = recompute_metrics(DATA_YAML, "val", P2_VAL_DIR, P2_VAL_DIR / "labels")
    payload = {
        "created_at": utc_now(),
        "status": "passed",
        "experiment_id": "P2-001",
        "checkpoint": str(checkpoint),
        "split": "val",
        "test_inference_run": False,
        "metrics_yolov10": {
            str(key): float(value) for key, value in metrics.results_dict.items()
        },
        "recomputed_metrics": coco,
    }
    write_json(manifest_path, payload)
    print(f"P2 validation diagnostic: {manifest_path}")
    return 0


def ensure_baseline_val_metrics() -> dict[str, Any]:
    summary_path = (
        STAGE3_EVIDENCE / "stage3_baseline_val_coco_metrics.json"
    )
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    labels = (
        PROJECT_ROOT
        / "03_实验"
        / "诊断"
        / "BASELINE-001"
        / "val_eval"
        / "labels"
    )
    if not labels.is_dir():
        raise FileNotFoundError(f"Baseline validation labels not found: {labels}")
    payload = recompute_metrics(
        DATA_YAML, "val", BASELINE_VAL_METRICS_DIR, labels
    )
    write_json(summary_path, payload)
    return payload


def analyze() -> int:
    names = load_names()
    class_stats = pd.read_csv(DATA_DIR / "stats" / "class_statistics.csv")
    p2_test_metrics = pd.read_csv(P2_ROOT / "test_eval" / "per_class_metrics.csv")
    p2_val_manifest = json.loads(
        (P2_VAL_DIR / "stage4_val_manifest.json").read_text(encoding="utf-8")
    )
    baseline_val_coco = ensure_baseline_val_metrics()

    test_gt, test_pred, _ = load_split(
        "test",
        DATA_DIR / "labels" / "test",
        P2_ROOT / "test_eval" / "labels",
        DATA_DIR / "images" / "test",
    )
    val_gt, val_pred, _ = load_split(
        "val",
        DATA_DIR / "labels" / "val",
        P2_VAL_DIR / "labels",
        DATA_DIR / "images" / "val",
    )
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
        p2_test_metrics,
        class_stats,
    )
    test_size = summarize_size_buckets(test_gt_events)
    val_size = summarize_size_buckets(val_gt_events)
    test_confusion = build_confusion_table(test_gt_events)

    comparison = pd.read_csv(
        BASELINE_ROOT / "test_eval" / "per_class_metrics.csv"
    ).merge(
        p2_test_metrics,
        on=["class_id", "class_name"],
        suffixes=("_baseline", "_p2"),
        how="outer",
    )
    for metric in ("precision", "recall", "map50", "map50_95"):
        comparison[f"delta_{metric}"] = (
            comparison[f"{metric}_p2"] - comparison[f"{metric}_baseline"]
        )

    baseline_summary = json.loads(
        (STAGE2_EVIDENCE / "stage2_test_evaluation.json").read_text(encoding="utf-8")
    )
    stage3_summary = json.loads(
        (
            PROJECT_ROOT
            / "03_实验"
            / "诊断"
            / "BASELINE-001"
            / "analysis"
            / "analysis_summary.json"
        ).read_text(encoding="utf-8")
    )
    p2_summary = json.loads(
        (STAGE4_EVIDENCE / "stage4_test_evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    baseline_training = json.loads(
        (STAGE2_EVIDENCE / "stage2_training_summary.json").read_text(
            encoding="utf-8"
        )
    )
    p2_results = pd.read_csv(P2_ROOT / "train" / "results.csv", skipinitialspace=True)
    p2_results.columns = [column.strip() for column in p2_results.columns]
    p2_results["fitness"] = (
        0.1 * p2_results["metrics/mAP50(B)"]
        + 0.9 * p2_results["metrics/mAP50-95(B)"]
    )
    p2_best = p2_results.loc[p2_results["fitness"].idxmax()]

    test_count = count_summary(test_gt_events, test_pred_events)
    val_count = count_summary(val_gt_events, val_pred_events)
    metric_comparison = {
        "baseline_test_map50": baseline_summary["recomputed_metrics"]["map50"],
        "p2_test_map50": p2_summary["recomputed_metrics"]["map50"],
        "delta_test_map50": (
            p2_summary["recomputed_metrics"]["map50"]
            - baseline_summary["recomputed_metrics"]["map50"]
        ),
        "baseline_test_map50_95": baseline_summary["recomputed_metrics"]["map50_95"],
        "p2_test_map50_95": p2_summary["recomputed_metrics"]["map50_95"],
        "delta_test_map50_95": (
            p2_summary["recomputed_metrics"]["map50_95"]
            - baseline_summary["recomputed_metrics"]["map50_95"]
        ),
        "baseline_test_map_small": baseline_summary["recomputed_metrics"]["map_small"],
        "p2_test_map_small": p2_summary["recomputed_metrics"]["map_small"],
        "delta_test_map_small": (
            p2_summary["recomputed_metrics"]["map_small"]
            - baseline_summary["recomputed_metrics"]["map_small"]
        ),
        "baseline_test_map_medium": baseline_summary["recomputed_metrics"]["map_medium"],
        "p2_test_map_medium": p2_summary["recomputed_metrics"]["map_medium"],
        "delta_test_map_medium": (
            p2_summary["recomputed_metrics"]["map_medium"]
            - baseline_summary["recomputed_metrics"]["map_medium"]
        ),
        "baseline_test_map_large": baseline_summary["recomputed_metrics"]["map_large"],
        "p2_test_map_large": p2_summary["recomputed_metrics"]["map_large"],
        "delta_test_map_large": (
            p2_summary["recomputed_metrics"]["map_large"]
            - baseline_summary["recomputed_metrics"]["map_large"]
        ),
        "baseline_val_map50": baseline_training["best_validation"]["map50"],
        "p2_val_map50": float(p2_best["metrics/mAP50(B)"]),
        "delta_val_map50": float(p2_best["metrics/mAP50(B)"])
        - baseline_training["best_validation"]["map50"],
        "baseline_val_map50_95": baseline_training["best_validation"]["map50_95"],
        "p2_val_map50_95": float(p2_best["metrics/mAP50-95(B)"]),
        "delta_val_map50_95": float(p2_best["metrics/mAP50-95(B)"])
        - baseline_training["best_validation"]["map50_95"],
        "baseline_val_best_epoch": baseline_training["best_epoch_csv"],
        "p2_val_best_epoch": int(p2_best["epoch"]),
        "baseline_val_map_small": baseline_val_coco["map_small"],
        "p2_val_map_small": p2_val_manifest["recomputed_metrics"]["map_small"],
        "delta_val_map_small": (
            p2_val_manifest["recomputed_metrics"]["map_small"]
            - baseline_val_coco["map_small"]
        ),
    }

    output = {
        "created_at": utc_now(),
        "experiment_id": "P2-001",
        "comparison_against": "BASELINE-001",
        "operational_confidence": OPERATIONAL_CONF,
        "test_inference_count": 1,
        "metric_comparison": metric_comparison,
        "test_operational": {
            "baseline": stage3_summary["test_summary"],
            "p2": test_count,
        },
        "val_operational": {
            "baseline": stage3_summary["val_summary"],
            "p2": val_count,
        },
        "test_size_buckets": test_size.to_dict(orient="records"),
        "val_size_buckets": val_size.to_dict(orient="records"),
    }
    write_json(STAGE4_EVIDENCE / "stage4_comparison.json", output)
    comparison.to_csv(
        STAGE4_EVIDENCE / "stage4_per_class_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_per_class.to_csv(
        STAGE4_EVIDENCE / "stage4_test_operational_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_confusion.to_csv(
        STAGE4_EVIDENCE / "stage4_test_confusion_pairs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    figure_path = STAGE4_EVIDENCE / "stage4_metric_comparison.png"
    labels_to_plot = ["mAP50", "mAP50-95", "mAP small", "mAP medium", "mAP large"]
    baseline_values = [
        metric_comparison["baseline_test_map50"],
        metric_comparison["baseline_test_map50_95"],
        metric_comparison["baseline_test_map_small"],
        metric_comparison["baseline_test_map_medium"],
        metric_comparison["baseline_test_map_large"],
    ]
    p2_values = [
        metric_comparison["p2_test_map50"],
        metric_comparison["p2_test_map50_95"],
        metric_comparison["p2_test_map_small"],
        metric_comparison["p2_test_map_medium"],
        metric_comparison["p2_test_map_large"],
    ]
    positions = range(len(labels_to_plot))
    fig, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar([x - 0.2 for x in positions], baseline_values, width=0.4, label="Baseline")
    axis.bar([x + 0.2 for x in positions], p2_values, width=0.4, label="P2-001")
    axis.set_xticks(list(positions), labels_to_plot)
    axis.set_ylim(0, 0.65)
    axis.set_ylabel("COCO metric")
    axis.set_title("BASELINE-001 versus P2-001 test metrics")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    output["figure"] = str(figure_path)
    write_json(STAGE4_EVIDENCE / "stage4_comparison.json", output)
    print(json.dumps(output["metric_comparison"], ensure_ascii=False, indent=2))
    print(json.dumps({"test_operational": output["test_operational"], "val_operational": output["val_operational"]}, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare-val", "analyze"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.command == "prepare-val":
        return prepare_val(args)
    return analyze()


if __name__ == "__main__":
    raise SystemExit(main())
