"""Recover the Stage 2 training summary after a post-training JSON bug.

The 40-epoch model training and Ultralytics artifact generation completed.
Only the runner's final summary serialization failed because argparse's
callable ``func`` field was included in ``vars(args)``. This script reconstructs
the summary from immutable training artifacts and records the recovery cause.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SOURCE_ROOT = PROJECT_ROOT / "02_代码" / "源码" / "yolov10"
EXP_ROOT = PROJECT_ROOT / "03_实验" / "baseline_yolov10n"
TRAIN_DIR = EXP_ROOT / "train"
WEIGHTS_DIR = TRAIN_DIR / "weights"
EVIDENCE_ROOT = PROJECT_ROOT / "01_立项" / "阶段2_证据"
LOG_PATH = PROJECT_ROOT / "03_实验" / "logs" / "BASELINE-001" / "train_stdout.log"
ARGS_PATH = TRAIN_DIR / "args.yaml"
RESULTS_CSV = TRAIN_DIR / "results.csv"
WEIGHT_EQUIVALENCE = EVIDENCE_ROOT / "权重等价性校验.json"
PARTIAL_SUMMARY = EVIDENCE_ROOT / "stage2_training_summary.partial_invalid.json"
RECOVERED_SUMMARY = EVIDENCE_ROOT / "stage2_training_summary.recovered.json"

sys.path.insert(0, str(SOURCE_ROOT))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        headers = [header.strip() for header in next(reader)]
        return [
            {header: value.strip() for header, value in zip(headers, row, strict=False)}
            for row in reader
        ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    log_text = LOG_PATH.read_text(encoding="utf-8", errors="replace")
    partial_text = PARTIAL_SUMMARY.read_text(encoding="utf-8", errors="replace")
    started_match = re.search(r'"started_at":\s*"([^"]+)"', partial_text)
    started_at = (
        started_match.group(1)
        if started_match
        else datetime.fromtimestamp(LOG_PATH.stat().st_ctime)
        .astimezone()
        .isoformat(timespec="seconds")
    )

    duration_match = re.search(
        r"40 epochs completed in ([0-9.]+) hours", log_text
    )
    if not duration_match:
        raise RuntimeError("Could not find Ultralytics 40-epoch completion line")
    duration_hours = float(duration_match.group(1))

    gpu_values = [
        float(value)
        for value in re.findall(r"\s([0-9]+(?:\.[0-9]+)?)G\s", log_text)
    ]
    reported_gpu_gib = max(gpu_values) if gpu_values else None

    rows = read_csv_rows(RESULTS_CSV)
    if len(rows) != 40:
        raise RuntimeError(f"Expected 40 result rows, found {len(rows)}")
    for row in rows:
        row["fitness_derived"] = 0.1 * float(
            row["metrics/mAP50(B)"]
        ) + 0.9 * float(row["metrics/mAP50-95(B)"])
    best_row = max(rows, key=lambda row: float(row["fitness_derived"]))

    train_args = yaml.safe_load(ARGS_PATH.read_text(encoding="utf-8"))
    checkpoint = torch.load(WEIGHTS_DIR / "best.pt", map_location="cpu")
    checkpoint_model = checkpoint.get("ema") or checkpoint["model"]
    parameter_count = sum(parameter.numel() for parameter in checkpoint_model.parameters())
    weight_equivalence = json.loads(
        WEIGHT_EQUIVALENCE.read_text(encoding="utf-8")
    )
    completion_evidence = {
        "completion_line": duration_match.group(0),
        "best_checkpoint_optimizer_stripped": (
            "Optimizer stripped from" in log_text
            and "best.pt" in log_text
        ),
        "results_saved_line": "Results saved to" in log_text,
        "traceback_present": "TypeError: Object of type function is not JSON serializable"
        in log_text,
    }
    plots = [
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "F1_curve.png",
        "P_curve.png",
        "PR_curve.png",
        "R_curve.png",
        "val_batch0_labels.jpg",
        "val_batch0_pred.jpg",
    ]
    plot_paths = {name: str(TRAIN_DIR / name) for name in plots}
    all_plots_exist = all((TRAIN_DIR / name).is_file() for name in plots)
    if not all(completion_evidence.values()) or not all_plots_exist:
        raise RuntimeError(
            f"Recovery preconditions failed: {completion_evidence}, plots={all_plots_exist}"
        )

    summary = {
        "experiment_id": "BASE_YOLOV10N_TT100K45_S2026",
        "started_at": started_at,
        "finished_at": datetime.fromtimestamp(
            LOG_PATH.stat().st_mtime
        ).astimezone().isoformat(timespec="seconds"),
        "status": "passed",
        "data_yaml": str(
            PROJECT_ROOT
            / "03_实验"
            / "datasets"
            / "TT100K"
            / "yolo"
            / "tt100k.yaml"
        ),
        "weights_hf": str(PROJECT_ROOT / "02_代码" / "权重" / "yolov10n_hf"),
        "trainable_weight_carrier": str(
            PROJECT_ROOT / "02_代码" / "权重" / "yolov10n.pt"
        ),
        "command_arguments": {
            key: train_args[key]
            for key in [
                "epochs",
                "batch",
                "imgsz",
                "optimizer",
                "patience",
                "close_mosaic",
                "save_period",
                "device",
                "workers",
                "seed",
            ]
        },
        "model": {
            "parameters": int(parameter_count),
            "source": "best.pt model after optimizer stripping",
        },
        "weight_equivalence": weight_equivalence,
        "completed_epochs": len(rows),
        "best_epoch_csv": int(best_row["epoch"]),
        "best_validation": {
            "fitness_derived": float(best_row["fitness_derived"]),
            "precision": float(best_row["metrics/precision(B)"]),
            "recall": float(best_row["metrics/recall(B)"]),
            "map50": float(best_row["metrics/mAP50(B)"]),
            "map50_95": float(best_row["metrics/mAP50-95(B)"]),
        },
        "training_duration_hours_reported": duration_hours,
        "elapsed_seconds": round(duration_hours * 3600.0, 3),
        "reported_max_gpu_mem_gib": reported_gpu_gib,
        "train_batches": 40 * 687,
        "train_dataset_images": 5489,
        "best_checkpoint": str(WEIGHTS_DIR / "best.pt"),
        "last_checkpoint": str(WEIGHTS_DIR / "last.pt"),
        "completion_evidence": completion_evidence,
        "plot_evidence": plot_paths,
        "recovery": {
            "required": True,
            "reason": (
                "训练主体和全部 Ultralytics 产物正常完成；最终 JSON 汇总因 "
                "argparse vars(args) 中的 func 为函数对象而序列化失败。"
            ),
            "source_partial_invalid": str(PARTIAL_SUMMARY),
            "source_log": str(LOG_PATH),
            "source_results_csv": str(RESULTS_CSV),
            "source_args": str(ARGS_PATH),
            "source_log_sha256": sha256(LOG_PATH),
            "source_results_sha256": sha256(RESULTS_CSV),
            "source_args_sha256": sha256(ARGS_PATH),
            "started_at_recovered_from_partial_prefix": started_match is not None,
        },
    }
    RECOVERED_SUMMARY.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Recovered training summary: {RECOVERED_SUMMARY}")
    print(f"Status: {summary['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
