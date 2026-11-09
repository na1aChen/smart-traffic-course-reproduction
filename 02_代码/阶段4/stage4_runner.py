"""Stage 4 runner for the independently switchable P2 experiment."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
STAGE2_CODE = PROJECT_ROOT / "02_代码" / "阶段2"
SOURCE_ROOT = PROJECT_ROOT / "02_代码" / "源码" / "yolov10"
DATA_YAML = (
    PROJECT_ROOT
    / "03_实验"
    / "datasets"
    / "TT100K"
    / "yolo"
    / "tt100k.yaml"
)
WEIGHTS_PT = PROJECT_ROOT / "02_代码" / "权重" / "yolov10n.pt"
BASELINE_YAML = (
    PROJECT_ROOT / "02_代码" / "阶段4" / "configs" / "yolov10n-45.yaml"
)
P2_YAML = (
    PROJECT_ROOT
    / "02_代码"
    / "阶段4"
    / "configs"
    / "yolov10n-p2-45.yaml"
)
EVIDENCE_ROOT = PROJECT_ROOT / "01_立项" / "阶段4_证据"
LOG_ROOT = PROJECT_ROOT / "03_实验" / "logs" / "P2-001"
EXP_ROOT = PROJECT_ROOT / "03_实验" / "P2-001"
SMOKE_ROOT = PROJECT_ROOT / "03_实验" / "P2-001" / "smoke"

sys.path.insert(0, str(STAGE2_CODE))
sys.path.insert(0, str(SOURCE_ROOT))

from stage2_runner import (  # noqa: E402
    _plain_metrics,
    _write_per_class_csv,
    Tee,
    get_flops,
    install_unicode_path_save_patch,
    model_metadata,
    recompute_metrics,
    register_memory_callbacks,
    set_reproducibility,
    tee_console,
    utc_now,
)
from ultralytics import YOLOv10  # noqa: E402


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def transfer_stats(model: YOLOv10, weights_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(weights_path, map_location="cpu")
    source = checkpoint["model"].float().state_dict()
    target = model.model.state_dict()
    matched_keys = [
        key
        for key in source
        if key in target and source[key].shape == target[key].shape
    ]
    matched_parameters = sum(source[key].numel() for key in matched_keys)
    unmatched_keys = [key for key in target if key not in matched_keys]
    unmatched_modules = Counter(
        ".".join(key.split(".")[:3]) for key in unmatched_keys
    )
    return {
        "source_tensor_count": len(source),
        "target_tensor_count": len(target),
        "matched_tensor_count": len(matched_keys),
        "matched_parameter_count": matched_parameters,
        "unmatched_target_tensor_count": len(target) - len(matched_keys),
        "unmatched_target_examples": unmatched_keys[:20],
        "unmatched_target_modules": dict(
            sorted(unmatched_modules.items(), key=lambda item: item[0])
        ),
    }


def build_model(variant: str) -> tuple[YOLOv10, dict[str, Any]]:
    yaml_path = BASELINE_YAML if variant == "baseline" else P2_YAML
    model = YOLOv10(str(yaml_path))
    transfer = transfer_stats(model, WEIGHTS_PT)
    model.load(str(WEIGHTS_PT))
    transfer["loaded_with"] = "Model.load(path)"
    return model, transfer


def model_fingerprint(variant: str, device: str = "0") -> dict[str, Any]:
    model, transfer = build_model(variant)
    metadata = model_metadata(model, imgsz=640)
    strides = [float(value) for value in model.model.stride.tolist()]
    head = model.model.model[-1]
    output_layers = getattr(head, "f", None)
    return {
        "variant": variant,
        "yaml": str(BASELINE_YAML if variant == "baseline" else P2_YAML),
        "classes": int(head.nc),
        "parameters": int(metadata["parameters"]),
        "gflops_640": metadata["flops_gflops_at_imgsz"],
        "flops_error": metadata["flops_error"],
        "detect_strides": strides,
        "detect_input_layers": output_layers,
        "weight_transfer": transfer,
        "device_requested": device,
    }


def inspect_models() -> int:
    install_unicode_path_save_patch()
    baseline = model_fingerprint("baseline")
    p2 = model_fingerprint("p2")
    parameter_delta = p2["parameters"] - baseline["parameters"]
    flops_delta = (
        p2["gflops_640"] - baseline["gflops_640"]
        if p2["gflops_640"] is not None and baseline["gflops_640"] is not None
        else None
    )
    payload = {
        "created_at": utc_now(),
        "status": "passed"
        if baseline["detect_strides"] == [8.0, 16.0, 32.0]
        and p2["detect_strides"] == [4.0, 8.0, 16.0, 32.0]
        and p2["classes"] == 45
        else "failed",
        "baseline": baseline,
        "p2": p2,
        "delta": {
            "parameters": parameter_delta,
            "parameters_percent": round(
                100.0 * parameter_delta / baseline["parameters"], 4
            )
            if baseline["parameters"]
            else None,
            "gflops_640": flops_delta,
        },
        "test_inference_run": False,
        "formal_training_run": False,
    }
    output_path = EVIDENCE_ROOT / "P2架构检查.json"
    write_json(output_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "passed" else 1


def smoke(args: argparse.Namespace) -> int:
    install_unicode_path_save_patch()
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = SMOKE_ROOT / f"run_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    summary_path = EVIDENCE_ROOT / "P2_smoke_summary.json"
    started = time.perf_counter()
    summary: dict[str, Any] = {
        "created_at": utc_now(),
        "experiment_id": "P2-001-SMOKE",
        "status": "started",
        "variant": args.variant,
        "yaml": str(P2_YAML),
        "weights": str(WEIGHTS_PT),
        "data_yaml": str(DATA_YAML),
        "run_dir": str(run_dir),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "fraction": args.fraction,
        "epochs": 1,
        "test_inference_run": False,
    }
    try:
        set_reproducibility(args.seed)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        model, transfer = build_model(args.variant)
        summary["weight_transfer"] = transfer
        summary["model_before_training"] = model_metadata(model, imgsz=args.imgsz)
        summary["detect_strides"] = [
            float(value) for value in model.model.stride.tolist()
        ]
        memory_state = register_memory_callbacks(model)
        loss_values: list[float] = []

        def on_train_batch_end(trainer) -> None:
            loss_items = getattr(trainer, "loss_items", None)
            if loss_items is not None:
                loss_values.extend(float(value) for value in loss_items)

        model.add_callback("on_train_batch_end", on_train_batch_end)
        with tee_console(run_dir / "console.log"):
            model.train(
                data=str(DATA_YAML),
                epochs=1,
                batch=args.batch,
                imgsz=args.imgsz,
                device=args.device,
                workers=0,
                optimizer="SGD",
                lr0=0.01,
                lrf=0.01,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=0.0,
                seed=args.seed,
                deterministic=True,
                amp=True,
                patience=1,
                close_mosaic=0,
                fraction=args.fraction,
                project=str(SMOKE_ROOT),
                name=run_dir.name,
                exist_ok=True,
                plots=False,
                save=False,
                val=False,
                cache=False,
                verbose=True,
            )
        finite_losses = all(math.isfinite(value) for value in loss_values)
        allocated = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        reserved = int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0
        summary.update(
            {
                "status": "passed"
                if memory_state.get("train_batches", 0) > 0
                and finite_losses
                and summary["detect_strides"] == [4.0, 8.0, 16.0, 32.0]
                else "failed",
                "train_batches": memory_state.get("train_batches", 0),
                "train_dataset_images": memory_state.get("train_dataset_images"),
                "train_epoch_seconds": memory_state.get("train_epoch_seconds"),
                "train_images_per_second": memory_state.get("train_images_per_second"),
                "peak_memory_allocated_bytes": allocated,
                "peak_memory_reserved_bytes": reserved,
                "peak_memory_allocated_gib": round(allocated / 1024**3, 4),
                "peak_memory_reserved_gib": round(reserved / 1024**3, 4),
                "loss_values_tail": loss_values[-10:],
                "losses_finite": finite_losses,
                "test_inference_run": False,
            }
        )
    except Exception as exc:  # noqa: BLE001
        summary.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "test_inference_run": False,
            }
        )
        error_log = run_dir / "error.log"
        error_log.write_text(summary["traceback"], encoding="utf-8")
    summary["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    summary["finished_at"] = utc_now()
    write_json(summary_path, summary)
    print(f"P2 smoke summary: {summary_path}")
    return 0 if summary["status"] == "passed" else 1


def validate_smoke_gate() -> dict[str, Any]:
    smoke_summary_path = EVIDENCE_ROOT / "P2_smoke_summary.json"
    if not smoke_summary_path.is_file():
        raise FileNotFoundError(f"P2 smoke summary not found: {smoke_summary_path}")
    smoke_summary = json.loads(smoke_summary_path.read_text(encoding="utf-8"))
    if smoke_summary.get("status") != "passed":
        raise RuntimeError("P2 smoke did not pass")
    if smoke_summary.get("detect_strides") != [4.0, 8.0, 16.0, 32.0]:
        raise RuntimeError("P2 smoke produced unexpected detection strides")
    if not smoke_summary.get("losses_finite"):
        raise RuntimeError("P2 smoke contained non-finite losses")
    if int(smoke_summary.get("batch", 0)) != 8:
        raise RuntimeError("P2 smoke was not run at the locked batch size 8")
    if int(smoke_summary.get("imgsz", 0)) != 640:
        raise RuntimeError("P2 smoke was not run at the locked input size 640")
    if int(smoke_summary.get("train_batches", 0)) <= 0:
        raise RuntimeError("P2 smoke did not execute a training batch")
    return smoke_summary


def count_results_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    return max(0, len(path.read_text(encoding="utf-8").splitlines()) - 1)


def find_resume_checkpoint() -> tuple[Path, list[dict[str, str]]]:
    """Prefer last.pt, but fall back to the newest loadable verified backup."""
    preferred = EXP_ROOT / "train" / "weights" / "last.pt"
    backup_dir = EXP_ROOT / "train" / "weights" / "backup"
    candidates = [preferred]
    if backup_dir.is_dir():
        candidates.extend(
            sorted(backup_dir.glob("last_*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
        )
    failures: list[dict[str, str]] = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            torch.load(candidate, map_location="cpu")
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {"path": str(candidate), "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        return candidate, failures
    raise FileNotFoundError("No loadable last.pt or checkpoint backup was found")


def train(args: argparse.Namespace) -> int:
    install_unicode_path_save_patch()
    smoke_summary = validate_smoke_gate()
    run_dir = EXP_ROOT / "train"
    if run_dir.exists():
        raise FileExistsError(
            f"Formal P2 run directory already exists and will not be overwritten: {run_dir}"
        )
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    run_status_path = LOG_ROOT / "run_status.json"
    summary_path = EVIDENCE_ROOT / "stage4_training_summary.json"
    started = time.perf_counter()
    summary: dict[str, Any] = {
        "created_at": utc_now(),
        "experiment_id": "P2-001",
        "status": "started",
        "variant": "p2",
        "yaml": str(P2_YAML),
        "weights": str(WEIGHTS_PT),
        "data_yaml": str(DATA_YAML),
        "run_dir": str(run_dir),
        "smoke_gate": {
            "status": smoke_summary["status"],
            "train_batches": smoke_summary["train_batches"],
            "peak_memory_reserved_gib": smoke_summary["peak_memory_reserved_gib"],
        },
        "args": {
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "optimizer": args.optimizer,
            "patience": args.patience,
            "close_mosaic": args.close_mosaic,
            "save_period": args.save_period,
            "device": args.device,
            "workers": args.workers,
            "seed": args.seed,
        },
        "test_inference_run": False,
    }
    write_json(
        run_status_path,
        {
            "experiment_id": "P2-001",
            "status": "running",
            "stage": "train",
            "started_at": summary["created_at"],
        },
    )
    try:
        set_reproducibility(args.seed)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        model, transfer = build_model("p2")
        summary["weight_transfer"] = transfer
        summary["model"] = model_metadata(model, imgsz=args.imgsz)
        summary["detect_strides"] = [
            float(value) for value in model.model.stride.tolist()
        ]
        memory_state = register_memory_callbacks(model)
        loss_values: list[float] = []

        def on_train_batch_end(trainer) -> None:
            loss_items = getattr(trainer, "loss_items", None)
            if loss_items is not None:
                loss_values.extend(float(value) for value in loss_items)

        model.add_callback("on_train_batch_end", on_train_batch_end)
        with tee_console(run_dir / "console.log"):
            model.train(
                data=str(DATA_YAML),
                epochs=args.epochs,
                batch=args.batch,
                imgsz=args.imgsz,
                device=args.device,
                workers=args.workers,
                optimizer=args.optimizer,
                lr0=0.01,
                lrf=0.01,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=3.0,
                warmup_bias_lr=0.1,
                nbs=64,
                seed=args.seed,
                deterministic=True,
                amp=True,
                patience=args.patience,
                close_mosaic=args.close_mosaic,
                save_period=args.save_period,
                plots=True,
                save=True,
                val=True,
                cache=False,
                project=str(EXP_ROOT),
                name="train",
                exist_ok=True,
                verbose=True,
            )
        best_checkpoint = run_dir / "weights" / "best.pt"
        last_checkpoint = run_dir / "weights" / "last.pt"
        results_csv = run_dir / "results.csv"
        finite_losses = all(math.isfinite(value) for value in loss_values)
        completed_epochs = count_results_rows(results_csv)
        summary.update(memory_state)
        summary.update(
            {
                "status": "passed"
                if finite_losses
                and completed_epochs > 0
                and best_checkpoint.is_file()
                and last_checkpoint.is_file()
                else "failed",
                "completed_epochs": completed_epochs,
                "losses_finite": finite_losses,
                "loss_values_tail": loss_values[-10:],
                "best_checkpoint": str(best_checkpoint),
                "last_checkpoint": str(last_checkpoint),
                "results_csv": str(results_csv),
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
                "traceback": traceback.format_exc(),
            }
        )
    summary["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    summary["finished_at"] = utc_now()
    write_json(summary_path, summary)
    write_json(
        run_status_path,
        {
            "experiment_id": "P2-001",
            "status": "failed" if summary["status"] != "passed" else "train_passed",
            "stage": "train",
            "training_summary": str(summary_path),
            "finished_at": summary["finished_at"],
            "error": summary.get("error"),
        },
    )
    print(f"P2 training summary: {summary_path}")
    return 0 if summary["status"] == "passed" else 1


def evaluate(args: argparse.Namespace) -> int:
    install_unicode_path_save_patch()
    checkpoint = EXP_ROOT / "train" / "weights" / "best.pt"
    eval_dir = EXP_ROOT / "test_eval"
    summary_path = EVIDENCE_ROOT / "stage4_test_evaluation.json"
    run_status_path = LOG_ROOT / "run_status.json"
    if eval_dir.exists() or summary_path.exists():
        raise FileExistsError(
            "Stage 4 test evaluation already exists; refusing to evaluate test again"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"P2 best checkpoint not found: {checkpoint}")
    started = time.perf_counter()
    summary: dict[str, Any] = {
        "created_at": utc_now(),
        "experiment_id": "P2-001",
        "status": "started",
        "checkpoint": str(checkpoint),
        "data_yaml": str(DATA_YAML),
        "split": "test",
        "selection_rule": "best.pt selected by validation fitness; test evaluated exactly once",
        "test_inference_run_count": 1,
    }
    try:
        set_reproducibility(args.seed)
        model = YOLOv10(str(checkpoint))
        summary["model"] = model_metadata(model, imgsz=args.imgsz)
        summary["detect_strides"] = [
            float(value) for value in model.model.stride.tolist()
        ]
        eval_dir.mkdir(parents=True, exist_ok=False)
        with tee_console(eval_dir / "console.log"):
            metrics = model.val(
                data=str(DATA_YAML),
                split="test",
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                project=str(EXP_ROOT),
                name="test_eval",
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
        plain_metrics, per_class = _plain_metrics(metrics, imgsz=args.imgsz)
        write_json(eval_dir / "metrics_yolov10.json", plain_metrics)
        _write_per_class_csv(eval_dir / "per_class_metrics.csv", per_class)
        summary["metrics_yolov10"] = plain_metrics
        summary["raw_prediction_labels"] = str(eval_dir / "labels")
        summary["checkpoint_bytes"] = checkpoint.stat().st_size
        summary["recomputed_metrics"] = recompute_metrics(
            DATA_YAML, "test", eval_dir, eval_dir / "labels"
        )
        summary["status"] = "passed"
    except Exception as exc:  # noqa: BLE001
        summary.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    summary["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    summary["finished_at"] = utc_now()
    write_json(summary_path, summary)
    write_json(
        run_status_path,
        {
            "experiment_id": "P2-001",
            "status": "failed" if summary["status"] != "passed" else "passed",
            "stage": "complete",
            "training_summary": str(EVIDENCE_ROOT / "stage4_training_summary.json"),
            "evaluation_summary": str(summary_path),
            "finished_at": summary["finished_at"],
            "error": summary.get("error"),
        },
    )
    print(f"P2 evaluation summary: {summary_path}")
    return 0 if summary["status"] == "passed" else 1


def resume(args: argparse.Namespace) -> int:
    """Resume P2-001 from the latest complete Ultralytics checkpoint."""
    install_unicode_path_save_patch()
    checkpoint, rejected_checkpoints = find_resume_checkpoint()
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    run_status_path = LOG_ROOT / "run_status.json"
    summary_path = EVIDENCE_ROOT / "stage4_training_summary.json"
    started = time.perf_counter()
    summary: dict[str, Any] = {
        "created_at": utc_now(),
        "experiment_id": "P2-001",
        "status": "started",
        "mode": "resume",
        "checkpoint": str(checkpoint),
        "rejected_checkpoints": rejected_checkpoints,
        "yaml": str(P2_YAML),
        "data_yaml": str(DATA_YAML),
        "test_inference_run": False,
    }
    write_json(
        run_status_path,
        {
            "experiment_id": "P2-001",
            "status": "running",
            "stage": "resume_train",
            "started_at": summary["created_at"],
        },
    )
    try:
        set_reproducibility(args.seed)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        model = YOLOv10(str(checkpoint))
        summary["detect_strides"] = [
            float(value) for value in model.model.stride.tolist()
        ]
        summary["model"] = model_metadata(model, imgsz=640)
        memory_state = register_memory_callbacks(model)
        loss_values: list[float] = []

        def on_train_batch_end(trainer) -> None:
            loss_items = getattr(trainer, "loss_items", None)
            if loss_items is not None:
                loss_values.extend(float(value) for value in loss_items)

        model.add_callback("on_train_batch_end", on_train_batch_end)
        model.train(resume=True)
        results_csv = EXP_ROOT / "train" / "results.csv"
        best_checkpoint = EXP_ROOT / "train" / "weights" / "best.pt"
        last_checkpoint = EXP_ROOT / "train" / "weights" / "last.pt"
        completed_epochs = count_results_rows(results_csv)
        finite_losses = all(math.isfinite(value) for value in loss_values)
        summary.update(memory_state)
        summary.update(
            {
                "status": "passed"
                if finite_losses
                and completed_epochs > 0
                and best_checkpoint.is_file()
                and last_checkpoint.is_file()
                else "failed",
                "completed_epochs": completed_epochs,
                "losses_finite": finite_losses,
                "loss_values_tail": loss_values[-10:],
                "best_checkpoint": str(best_checkpoint),
                "last_checkpoint": str(last_checkpoint),
                "results_csv": str(results_csv),
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
                "traceback": traceback.format_exc(),
            }
        )
    summary["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    summary["finished_at"] = utc_now()
    write_json(summary_path, summary)
    write_json(
        run_status_path,
        {
            "experiment_id": "P2-001",
            "status": "failed" if summary["status"] != "passed" else "train_passed",
            "stage": "train",
            "training_summary": str(summary_path),
            "finished_at": summary["finished_at"],
            "error": summary.get("error"),
        },
    )
    print(f"P2 resume summary: {summary_path}")
    return 0 if summary["status"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("inspect", "smoke", "train", "resume", "evaluate")
    )
    parser.add_argument("--variant", choices=("baseline", "p2"), default="p2")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--fraction", type=float, default=0.02)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--close-mosaic", type=int, default=5)
    parser.add_argument("--save-period", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if args.command == "inspect":
        return inspect_models()
    if args.command == "smoke":
        return smoke(args)
    if args.command == "train":
        return train(args)
    if args.command == "resume":
        return resume(args)
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
