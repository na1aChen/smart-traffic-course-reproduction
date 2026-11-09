"""Sequential robustness and training-budget experiments for Stage 5."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
STAGE2_CODE = PROJECT_ROOT / "02_代码" / "阶段2"
STAGE4_CODE = PROJECT_ROOT / "02_代码" / "阶段4"
SOURCE_ROOT = PROJECT_ROOT / "02_代码" / "源码" / "yolov10"
DATA_YAML = (
    PROJECT_ROOT
    / "03_实验"
    / "datasets"
    / "TT100K"
    / "yolo"
    / "tt100k.yaml"
)
CONFIG_ROOT = PROJECT_ROOT / "03_实验" / "configs"
RUN_ROOT = PROJECT_ROOT / "03_实验" / "robustness"
LOG_ROOT = PROJECT_ROOT / "03_实验" / "logs" / "STAGE5"
EVIDENCE_ROOT = PROJECT_ROOT / "01_立项" / "阶段5_证据"

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "BASELINE-002-S2027": {
        "config": CONFIG_ROOT / "BASELINE-002.yaml",
        "variant": "baseline",
    },
    "P2-002-S2027": {
        "config": CONFIG_ROOT / "P2-002.yaml",
        "variant": "p2",
    },
    "BASELINE-BUDGET48-S2026": {
        "config": CONFIG_ROOT / "BASELINE-BUDGET48.yaml",
        "variant": "baseline",
    },
}

sys.path.insert(0, str(STAGE2_CODE))
sys.path.insert(0, str(STAGE4_CODE))
sys.path.insert(0, str(SOURCE_ROOT))

from stage2_runner import (  # noqa: E402
    _plain_metrics,
    _write_per_class_csv,
    install_unicode_path_save_patch,
    model_metadata,
    recompute_metrics,
    register_memory_callbacks,
    set_reproducibility,
    tee_console,
)
from stage4_runner import build_model  # noqa: E402
from ultralytics import YOLOv10  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_experiment(experiment_id: str) -> tuple[dict[str, Any], str]:
    if experiment_id not in EXPERIMENTS:
        raise KeyError(f"Unknown Stage 5 experiment: {experiment_id}")
    definition = EXPERIMENTS[experiment_id]
    config = yaml.safe_load(
        Path(definition["config"]).read_text(encoding="utf-8")
    )
    if config["experiment_id"] != experiment_id:
        raise ValueError(
            f"Config ID mismatch: {config['experiment_id']} != {experiment_id}"
        )
    return config, str(definition["variant"])


def result_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header = [value.strip() for value in rows[0]]
    return [
        dict(zip(header, [value.strip() for value in row]))
        for row in rows[1:]
        if row
    ]


def train_experiment(experiment_id: str) -> int:
    config, variant = load_experiment(experiment_id)
    exp_root = RUN_ROOT / experiment_id
    run_dir = exp_root / "train"
    if exp_root.exists():
        raise FileExistsError(f"Stage 5 experiment already exists: {exp_root}")
    log_dir = LOG_ROOT / experiment_id
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = EVIDENCE_ROOT / "training" / f"{experiment_id}.json"
    status_path = log_dir / "status.json"
    started_at = utc_now()
    summary: dict[str, Any] = {
        "created_at": started_at,
        "experiment_id": experiment_id,
        "config": str(EXPERIMENTS[experiment_id]["config"]),
        "variant": variant,
        "status": "running",
        "test_inference_run": False,
    }
    write_json(
        status_path,
        {"experiment_id": experiment_id, "status": "training", "started_at": started_at},
    )
    try:
        set_reproducibility(int(config["seed"]))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        model, transfer = build_model(variant)
        summary["weight_transfer"] = transfer
        summary["model"] = model_metadata(model, imgsz=int(config["imgsz"]))
        memory_state = register_memory_callbacks(model)
        losses: list[float] = []

        def on_train_batch_end(trainer) -> None:
            if getattr(trainer, "loss_items", None) is not None:
                losses.extend(float(value) for value in trainer.loss_items)

        model.add_callback("on_train_batch_end", on_train_batch_end)
        with tee_console(run_dir / "console.log"):
            model.train(
                data=str(DATA_YAML),
                epochs=int(config["epochs"]),
                batch=int(config["batch"]),
                imgsz=int(config["imgsz"]),
                device=str(config["device"]),
                workers=int(config["workers"]),
                optimizer=str(config["optimizer"]),
                lr0=float(config["lr0"]),
                lrf=float(config["lrf"]),
                momentum=float(config["momentum"]),
                weight_decay=float(config["weight_decay"]),
                warmup_epochs=float(config["warmup_epochs"]),
                nbs=int(config["nbs"]),
                seed=int(config["seed"]),
                deterministic=bool(config["deterministic"]),
                amp=bool(config["amp"]),
                patience=int(config["patience"]),
                close_mosaic=int(config["close_mosaic"]),
                save_period=5,
                plots=True,
                save=True,
                val=True,
                cache=False,
                project=str(exp_root),
                name="train",
                exist_ok=True,
                verbose=True,
            )
        rows = result_rows(run_dir / "results.csv")
        if rows:
            best = max(
                rows,
                key=lambda row: float(row["metrics/mAP50-95(B)"])
                + 0.1 * float(row["metrics/mAP50(B)"]),
            )
            best_epoch = int(best["epoch"])
            best_metrics = {
                "precision": float(best["metrics/precision(B)"]),
                "recall": float(best["metrics/recall(B)"]),
                "map50": float(best["metrics/mAP50(B)"]),
                "map50_95": float(best["metrics/mAP50-95(B)"]),
            }
        else:
            best_epoch = None
            best_metrics = None
        finite = all(math.isfinite(loss) for loss in losses)
        best_checkpoint = run_dir / "weights" / "best.pt"
        last_checkpoint = run_dir / "weights" / "last.pt"
        summary.update(memory_state)
        summary.update(
            {
                "status": "passed"
                if finite
                and rows
                and best_checkpoint.is_file()
                and last_checkpoint.is_file()
                else "failed",
                "completed_epochs": len(rows),
                "best_epoch": best_epoch,
                "best_validation": best_metrics,
                "losses_finite": finite,
                "results_csv": str(run_dir / "results.csv"),
                "best_checkpoint": str(best_checkpoint),
                "last_checkpoint": str(last_checkpoint),
                "peak_memory_allocated_gib": round(
                    memory_state.get("max_memory_allocated_bytes", 0) / 1024**3,
                    4,
                ),
                "peak_memory_reserved_gib": round(
                    memory_state.get("max_memory_reserved_bytes", 0) / 1024**3,
                    4,
                ),
            }
        )
    except Exception as exc:  # noqa: BLE001
        summary.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    summary["finished_at"] = utc_now()
    write_json(summary_path, summary)
    write_json(
        status_path,
        {
            "experiment_id": experiment_id,
            "status": "train_passed" if summary["status"] == "passed" else "failed",
            "training_summary": str(summary_path),
            "finished_at": summary["finished_at"],
        },
    )
    print(f"Stage 5 training summary: {summary_path}")
    return 0 if summary["status"] == "passed" else 1


def evaluate_experiment(experiment_id: str) -> int:
    config, _ = load_experiment(experiment_id)
    exp_root = RUN_ROOT / experiment_id
    checkpoint = exp_root / "train" / "weights" / "best.pt"
    eval_dir = exp_root / "test_eval"
    summary_path = EVIDENCE_ROOT / "evaluation" / f"{experiment_id}.json"
    log_dir = LOG_ROOT / experiment_id
    status_path = log_dir / "status.json"
    if eval_dir.exists() or summary_path.exists():
        raise FileExistsError(
            f"Test evaluation already exists for {experiment_id}; refusing rerun"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint}")
    started_at = utc_now()
    summary: dict[str, Any] = {
        "created_at": started_at,
        "experiment_id": experiment_id,
        "status": "running",
        "checkpoint": str(checkpoint),
        "split": "test",
        "test_inference_run_count": 1,
    }
    try:
        set_reproducibility(int(config["seed"]))
        model = YOLOv10(str(checkpoint))
        eval_dir.mkdir(parents=True, exist_ok=False)
        with tee_console(eval_dir / "console.log"):
            metrics = model.val(
                data=str(DATA_YAML),
                split="test",
                imgsz=int(config["imgsz"]),
                batch=int(config["batch"]),
                device=str(config["device"]),
                workers=int(config["workers"]),
                project=str(exp_root),
                name="test_eval",
                exist_ok=True,
                save_json=False,
                save_txt=True,
                save_conf=True,
                plots=True,
                conf=0.001,
                iou=0.7,
                max_det=100,
                seed=int(config["seed"]),
                deterministic=True,
                verbose=True,
            )
        plain, per_class = _plain_metrics(
            metrics, imgsz=int(config["imgsz"])
        )
        write_json(eval_dir / "metrics_yolov10.json", plain)
        _write_per_class_csv(eval_dir / "per_class_metrics.csv", per_class)
        summary["metrics_yolov10"] = plain
        summary["recomputed_metrics"] = recompute_metrics(
            DATA_YAML, "test", eval_dir, eval_dir / "labels"
        )
        summary["status"] = "passed"
    except Exception as exc:  # noqa: BLE001
        summary.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    summary["finished_at"] = utc_now()
    write_json(summary_path, summary)
    write_json(
        status_path,
        {
            "experiment_id": experiment_id,
            "status": "passed" if summary["status"] == "passed" else "failed",
            "evaluation_summary": str(summary_path),
            "finished_at": summary["finished_at"],
        },
    )
    print(f"Stage 5 evaluation summary: {summary_path}")
    return 0 if summary["status"] == "passed" else 1


def main() -> int:
    install_unicode_path_save_patch()
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), required=True)
    args = parser.parse_args()
    if args.command == "train":
        return train_experiment(args.experiment)
    return evaluate_experiment(args.experiment)


if __name__ == "__main__":
    raise SystemExit(main())
