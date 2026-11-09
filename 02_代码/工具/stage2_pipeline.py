#!/usr/bin/env python
"""Stage 2 baseline benchmark, training, prediction, and COCO evaluation."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "02_代码" / "源码" / "yolov10"
DATA_ROOT = PROJECT_ROOT / "03_实验" / "datasets" / "TT100K" / "yolo"
WEIGHTS = PROJECT_ROOT / "02_代码" / "权重" / "yolov10n.pt"
RUNS_ROOT = PROJECT_ROOT / "03_实验" / "runs"
PREDICTIONS_ROOT = PROJECT_ROOT / "03_实验" / "predictions"
METRICS_ROOT = PROJECT_ROOT / "03_实验" / "metrics"
COCO_ROOT = PROJECT_ROOT / "03_实验" / "annotations"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_dataset_config() -> dict[str, Any]:
    config = load_yaml(DATA_ROOT / "tt100k.yaml")
    names = config["names"]
    assert len(names) == 45, f"expected 45 classes, got {len(names)}"
    assert len(names) == len(set(names)), "duplicate class names in tt100k.yaml"
    return config


def validate_locked_inputs() -> dict[str, Any]:
    required = [
        WEIGHTS,
        DATA_ROOT / "tt100k.yaml",
        DATA_ROOT / "dataset_manifest.json",
        DATA_ROOT / "images" / "train",
        DATA_ROOT / "images" / "val",
        DATA_ROOT / "images" / "test",
        DATA_ROOT / "labels" / "train",
        DATA_ROOT / "labels" / "val",
        DATA_ROOT / "labels" / "test",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing locked inputs: {missing}")

    dataset_config = load_dataset_config()
    counts = {}
    for split in ("train", "val", "test"):
        images = sorted((DATA_ROOT / "images" / split).glob("*.jpg"))
        labels = sorted((DATA_ROOT / "labels" / split).glob("*.txt"))
        if len(images) != len(labels):
            raise ValueError(
                f"{split}: image/label count mismatch {len(images)} != {len(labels)}"
            )
        counts[split] = {"images": len(images), "labels": len(labels)}

    return {
        "checked_at": now_iso(),
        "classes": len(dataset_config["names"]),
        "split_counts": counts,
        "weights": {
            "path": str(WEIGHTS),
            "size_bytes": WEIGHTS.stat().st_size,
            "sha256": sha256_file(WEIGHTS),
        },
        "dataset_manifest_sha256": sha256_file(DATA_ROOT / "dataset_manifest.json"),
        "status": "passed",
    }


def benchmark_resources(output: Path) -> dict[str, Any]:
    """Measure synthetic train-step cost for the planned input/batch matrix."""

    from ultralytics import YOLOv10
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import YOLOv10DetectionModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the resource benchmark")

    torch.cuda.init()
    device = torch.device("cuda:0")
    pretrained = YOLOv10(str(WEIGHTS)).model
    model_cfg = pretrained.yaml
    del pretrained
    gc.collect()
    torch.cuda.empty_cache()

    cases = [
        {"imgsz": 512, "batch": 4},
        {"imgsz": 512, "batch": 8},
        {"imgsz": 640, "batch": 4},
        {"imgsz": 640, "batch": 8},
    ]
    results = []

    for case in cases:
        imgsz = int(case["imgsz"])
        batch_size = int(case["batch"])
        record: dict[str, Any] = {
            **case,
            "status": "started",
            "warmup_steps": 1,
            "measured_steps": 3,
            "synthetic_boxes_per_image": 3,
        }
        model = None
        optimizer = None
        scaler = None
        batch: dict[str, torch.Tensor] | None = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            model = YOLOv10DetectionModel(model_cfg, nc=45, verbose=False)
            model.load({"model": YOLOv10(str(WEIGHTS)).model})
            model.args = get_cfg()
            model = model.to(device).train()
            optimizer = torch.optim.SGD(
                model.parameters(), lr=0.01, momentum=0.937, weight_decay=0.0005
            )
            scaler = torch.cuda.amp.GradScaler(enabled=True)

            count = batch_size * int(record["synthetic_boxes_per_image"])
            batch_idx = torch.arange(batch_size, device=device).repeat_interleave(
                int(record["synthetic_boxes_per_image"])
            )
            cls = torch.arange(count, device=device).remainder(45).float().view(-1, 1)
            center = 0.2 + 0.6 * torch.rand((count, 2), device=device)
            size = 0.02 + 0.10 * torch.rand((count, 2), device=device)
            bboxes = torch.cat((center, size), dim=1)
            batch = {
                "img": torch.rand(
                    (batch_size, 3, imgsz, imgsz), device=device, dtype=torch.float32
                )
                * 255.0,
                "cls": cls,
                "bboxes": bboxes,
                "batch_idx": batch_idx,
            }

            def one_step(
                current_model: torch.nn.Module,
                current_optimizer: torch.optim.Optimizer,
                current_scaler: torch.cuda.amp.GradScaler,
                current_batch: dict[str, torch.Tensor],
            ) -> float:
                current_optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=True):
                    loss, _ = current_model(current_batch)
                current_scaler.scale(loss).backward()
                current_scaler.step(current_optimizer)
                current_scaler.update()
                return float(loss.detach())

            assert optimizer is not None and scaler is not None and batch is not None
            one_step(model, optimizer, scaler, batch)
            torch.cuda.synchronize(device)

            step_seconds = []
            losses = []
            for _ in range(int(record["measured_steps"])):
                started = time.perf_counter()
                loss_value = one_step(model, optimizer, scaler, batch)
                torch.cuda.synchronize(device)
                step_seconds.append(time.perf_counter() - started)
                losses.append(loss_value)

            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            record.update(
                {
                    "status": "passed",
                    "mean_step_seconds": float(np.mean(step_seconds)),
                    "std_step_seconds": float(np.std(step_seconds, ddof=0)),
                    "images_per_second": float(batch_size / np.mean(step_seconds)),
                    "losses": losses,
                    "peak_allocated_mib": torch.cuda.max_memory_allocated(device)
                    / (1024**2),
                    "peak_reserved_mib": torch.cuda.max_memory_reserved(device)
                    / (1024**2),
                    "device_total_mib": total_bytes / (1024**2),
                    "device_free_after_step_mib": free_bytes / (1024**2),
                }
            )
        except torch.cuda.OutOfMemoryError as exc:
            record.update({"status": "oom", "error": repr(exc)})
        finally:
            del batch, optimizer, scaler, model
            gc.collect()
            torch.cuda.empty_cache()
        results.append(record)

    payload = {
        "recorded_at": now_iso(),
        "purpose": "synthetic training-step resource measurement; not a model result",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "amp": True,
        "optimizer": "SGD",
        "results": results,
    }
    write_json(output, payload)
    return payload


def _train_payload(
    config: dict[str, Any], smoke: bool, run_name: str | None
) -> dict[str, Any]:
    payload = {
        "data": str(DATA_ROOT / "tt100k.yaml"),
        "epochs": int(config["epochs"]),
        "imgsz": int(config["imgsz"]),
        "batch": int(config["batch"]),
        "device": str(config["device"]),
        "workers": int(config["workers"]),
        "project": str(RUNS_ROOT),
        "name": run_name or str(config["experiment_id"]),
        "exist_ok": True,
        "pretrained": True,
        "optimizer": str(config["optimizer"]),
        "lr0": float(config["lr0"]),
        "lrf": float(config["lrf"]),
        "momentum": float(config["momentum"]),
        "weight_decay": float(config["weight_decay"]),
        "warmup_epochs": float(config["warmup_epochs"]),
        "warmup_bias_lr": float(config["warmup_bias_lr"]),
        "nbs": int(config["nbs"]),
        "patience": int(config["patience"]),
        "seed": int(config["seed"]),
        "deterministic": bool(config["deterministic"]),
        "amp": bool(config["amp"]),
        "cache": False,
        "rect": False,
        "cos_lr": bool(config.get("cos_lr", False)),
        "close_mosaic": int(config["close_mosaic"]),
        "val": True,
        "val_period": 1,
        "save": True,
        "save_period": -1,
        "plots": True,
        "fraction": 1.0,
        "single_cls": False,
        "resume": False,
    }
    if smoke:
        payload.update(
            {
                "epochs": 1,
                "fraction": 0.02,
                "patience": 0,
                "close_mosaic": 0,
                "name": run_name or "SMOKE-001",
            }
        )
    return payload


def train_baseline(config_path: Path, smoke: bool = False) -> dict[str, Any]:
    from ultralytics import YOLOv10

    config = load_yaml(config_path)
    validation = validate_locked_inputs()
    train_payload = _train_payload(config, smoke=smoke, run_name=None)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; baseline training is blocked")

    torch.cuda.init()
    model = YOLOv10(str(WEIGHTS))
    epoch_started: dict[int, float] = {}
    epochs: list[dict[str, Any]] = []

    def on_train_start(trainer: Any) -> None:
        torch.cuda.reset_peak_memory_stats(0)

    def on_train_epoch_start(trainer: Any) -> None:
        epoch_started[int(trainer.epoch)] = time.perf_counter()

    def on_train_epoch_end(trainer: Any) -> None:
        epoch = int(trainer.epoch)
        lrs = [
            float(group["lr"])
            for group in trainer.optimizer.param_groups
            if "lr" in group
        ]
        epochs.append(
            {
                "epoch_one_based": epoch + 1,
                "elapsed_seconds": time.perf_counter()
                - epoch_started.get(epoch, time.perf_counter()),
                "learning_rates": lrs,
                "peak_allocated_mib": torch.cuda.max_memory_allocated(0) / (1024**2),
                "peak_reserved_mib": torch.cuda.max_memory_reserved(0) / (1024**2),
            }
        )

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    started = time.perf_counter()
    failure: str | None = None
    try:
        model.train(**train_payload)
    except Exception as exc:
        failure = repr(exc)
        raise
    finally:
        torch.cuda.synchronize(0)
        total_seconds = time.perf_counter() - started
        trainer = getattr(model, "trainer", None)
        save_dir = Path(getattr(trainer, "save_dir", RUNS_ROOT / train_payload["name"]))
        metrics = {}
        if trainer is not None and getattr(trainer, "metrics", None):
            metrics = {
                key: float(value)
                for key, value in trainer.metrics.items()
                if isinstance(value, (int, float))
            }
        payload = {
            "experiment_id": train_payload["name"],
            "mode": "smoke" if smoke else "full_baseline",
            "recorded_at": now_iso(),
            "status": "failed" if failure else "passed",
            "failure": failure,
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "weights_path": str(WEIGHTS),
            "weights_sha256": validation["weights"]["sha256"],
            "training_args": train_payload,
            "data_validation": validation,
            "run_dir": str(save_dir),
            "best_checkpoint": str(save_dir / "weights" / "best.pt"),
            "last_checkpoint": str(save_dir / "weights" / "last.pt"),
            "results_csv": str(save_dir / "results.csv"),
            "results_png": str(save_dir / "results.png"),
            "total_seconds": total_seconds,
            "minutes_per_epoch": total_seconds / max(len(epochs), 1) / 60,
            "epochs": epochs,
            "final_train_metrics": metrics,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(0) / (1024**2),
            "peak_reserved_mib": torch.cuda.max_memory_reserved(0) / (1024**2),
            "effective_accumulate": int(getattr(trainer, "accumulate", 0) or 0),
        }
        write_json(save_dir / "resource_metrics.json", payload)
    return payload


def export_coco_ground_truth(split: str, output_dir: Path) -> dict[str, Any]:
    config = load_dataset_config()
    names = config["names"]
    image_dir = DATA_ROOT / "images" / split
    label_dir = DATA_ROOT / "labels" / split
    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"no {split} images in {image_dir}")

    image_records = []
    annotation_records = []
    image_ids: set[int] = set()
    annotation_id = 1
    for image_path in images:
        stem = image_path.stem
        if not stem.isnumeric():
            raise ValueError(f"non-numeric image id is unsupported: {image_path.name}")
        image_id = int(stem)
        if image_id in image_ids:
            raise ValueError(f"duplicate image id: {image_id}")
        image_ids.add(image_id)
        with Image.open(image_path) as image:
            width, height = image.size
        image_records.append(
            {
                "id": image_id,
                "file_name": f"{split}/{image_path.name}",
                "width": width,
                "height": height,
            }
        )

        label_path = label_dir / f"{stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"missing label: {label_path}")
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"{label_path}:{line_number} has {len(fields)} fields")
            class_id = int(fields[0])
            cx, cy, bw, bh = (float(value) for value in fields[1:])
            x = (cx - bw / 2) * width
            y = (cy - bh / 2) * height
            box_width = bw * width
            box_height = bh * height
            annotation_records.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": [x, y, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            annotation_id += 1

    payload = {
        "info": {
            "description": f"TT100K type45 {split} ground truth",
            "version": "stage2-coco-v1",
            "date_created": now_iso(),
        },
        "licenses": [{"id": 1, "name": "CC BY-NC"}],
        "images": image_records,
        "annotations": annotation_records,
        "categories": [
            {"id": index + 1, "name": name, "supercategory": "traffic_sign"}
            for index, name in enumerate(names)
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"instances_{split}.json"
    write_json(output_path, payload)
    manifest = {
        "recorded_at": now_iso(),
        "split": split,
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "classes": len(names),
        "images": len(image_records),
        "annotations": len(annotation_records),
    }
    write_json(output_dir / f"instances_{split}_manifest.json", manifest)
    return manifest


def predict_test(
    checkpoint: Path, experiment_id: str, batch: int = 8, imgsz: int = 640
) -> dict[str, Any]:
    from ultralytics import YOLOv10

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    image_paths = sorted((DATA_ROOT / "images" / "test").glob("*.jpg"))
    if len(image_paths) != 3071:
        raise ValueError(f"expected 3071 test images, got {len(image_paths)}")

    output_dir = PREDICTIONS_ROOT / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.init()
    model = YOLOv10(str(checkpoint))
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    detections = []
    per_image_counts: dict[int, int] = {}
    results = model.predict(
        source=[str(path) for path in image_paths],
        stream=True,
        imgsz=imgsz,
        conf=0.001,
        iou=0.7,
        max_det=300,
        device="0",
        batch=batch,
        half=True,
        save=False,
        verbose=False,
    )
    seen: set[int] = set()
    for result in results:
        image_id = int(Path(result.path).stem)
        if image_id in seen:
            raise ValueError(f"duplicate prediction result for image {image_id}")
        seen.add(image_id)
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            per_image_counts[image_id] = 0
            continue
        xyxy = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        for coordinates, score, class_id in zip(xyxy, scores, classes):
            x1, y1, x2, y2 = (float(value) for value in coordinates)
            detections.append(
                {
                    "image_id": image_id,
                    "category_id": int(class_id) + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )
        per_image_counts[image_id] = len(boxes)
    torch.cuda.synchronize(0)
    elapsed = time.perf_counter() - started
    missing = {int(path.stem) for path in image_paths} - seen
    if missing:
        raise RuntimeError(f"missing predictions for {len(missing)} images")

    prediction_path = output_dir / "predictions_coco.json"
    write_json(prediction_path, detections)
    manifest = {
        "recorded_at": now_iso(),
        "experiment_id": experiment_id,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "split": "test",
        "images": len(seen),
        "images_without_detections": sum(
            count == 0 for count in per_image_counts.values()
        ),
        "detections": len(detections),
        "mean_detections_per_image": len(detections) / len(seen),
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "settings": {
            "imgsz": imgsz,
            "batch": batch,
            "conf": 0.001,
            "iou": 0.7,
            "max_det": 300,
            "half": True,
        },
        "elapsed_seconds": elapsed,
        "images_per_second_end_to_end": len(seen) / elapsed,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(0) / (1024**2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(0) / (1024**2),
    }
    write_json(output_dir / "prediction_manifest.json", manifest)
    return manifest


def evaluate_coco(
    ground_truth: Path, predictions: Path, experiment_id: str
) -> dict[str, Any]:
    output_dir = METRICS_ROOT / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation = COCO(str(ground_truth))
    result = annotation.loadRes(str(predictions))
    image_ids = sorted(annotation.getImgIds())

    evaluator = COCOeval(annotation, result, "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    stats = [float(value) for value in evaluator.stats]
    metric_names = [
        "AP_IoU_0.50_0.95",
        "AP_IoU_0.50",
        "AP_IoU_0.75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR_maxDets_1",
        "AR_maxDets_10",
        "AR_maxDets_100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    metrics = dict(zip(metric_names, stats))

    per_category = []
    category_names = {
        category["id"]: category["name"]
        for category in annotation.loadCats(annotation.getCatIds())
    }
    for category_id in sorted(category_names):
        category_eval = COCOeval(annotation, result, "bbox")
        category_eval.params.imgIds = image_ids
        category_eval.params.catIds = [category_id]
        category_eval.params.maxDets = [1, 10, 100]
        category_eval.evaluate()
        category_eval.accumulate()
        category_eval.summarize()
        per_category.append(
            {
                "category_id": category_id,
                "class_name": category_names[category_id],
                "AP_IoU_0.50_0.95": float(category_eval.stats[0]),
                "AP_IoU_0.50": float(category_eval.stats[1]),
                "AP_IoU_0.75": float(category_eval.stats[2]),
                "AR_maxDets_100": float(category_eval.stats[8]),
            }
        )

    payload = {
        "recorded_at": now_iso(),
        "experiment_id": experiment_id,
        "ground_truth": str(ground_truth.resolve()),
        "ground_truth_sha256": sha256_file(ground_truth),
        "predictions": str(predictions.resolve()),
        "predictions_sha256": sha256_file(predictions),
        "images": len(image_ids),
        "metric_space": "COCO bbox AP in [0,1]",
        "metrics": metrics,
        "per_category": per_category,
    }
    write_json(output_dir / "coco_metrics.json", payload)

    with (output_dir / "per_class_ap.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_category[0]))
        writer.writeheader()
        writer.writerows(per_category)

    return payload


def validate_with_ultralytics(
    checkpoint: Path, experiment_id: str, batch: int = 8, imgsz: int = 640
) -> dict[str, Any]:
    from ultralytics import YOLOv10

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    output_dir = METRICS_ROOT / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLOv10(str(checkpoint))
    started = time.perf_counter()
    validator = model.val(
        data=str(DATA_ROOT / "tt100k.yaml"),
        split="test",
        imgsz=imgsz,
        batch=batch,
        device="0",
        workers=4,
        conf=0.001,
        iou=0.7,
        max_det=300,
        half=True,
        plots=True,
        save_json=False,
        save_txt=True,
        save_conf=True,
        project=str(METRICS_ROOT),
        name=f"{experiment_id}_ultralytics_test",
        exist_ok=True,
        verbose=True,
    )
    elapsed = time.perf_counter() - started
    results_dict = {
        key: float(value)
        for key, value in validator.metrics.results_dict.items()
        if isinstance(value, (int, float))
    }
    box_metrics = validator.metrics.box
    per_class = []
    names = validator.metrics.names
    for class_id in range(len(names)):
        per_class.append(
            {
                "category_id": class_id + 1,
                "class_name": names[class_id],
                "precision": float(box_metrics.p[class_id]),
                "recall": float(box_metrics.r[class_id]),
                "AP_IoU_0.50": float(box_metrics.ap50[class_id]),
                "AP_IoU_0.50_0.95": float(box_metrics.ap[class_id]),
            }
        )
    payload = {
        "recorded_at": now_iso(),
        "experiment_id": experiment_id,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "split": "test",
        "elapsed_seconds": elapsed,
        "results_dict": results_dict,
        "per_class": per_class,
        "save_dir": str(Path(validator.save_dir)),
    }
    write_json(output_dir / "ultralytics_test_metrics.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "03_实验"
        / "logs"
        / "BASELINE-001"
        / "input_validation.json",
    )

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "03_实验"
        / "logs"
        / "BASELINE-001"
        / "resource_benchmark.json",
    )

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--smoke", action="store_true")

    gt_parser = subparsers.add_parser("export-gt")
    gt_parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    gt_parser.add_argument("--output-dir", type=Path, default=COCO_ROOT)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--experiment-id", required=True)
    predict_parser.add_argument("--batch", type=int, default=8)
    predict_parser.add_argument("--imgsz", type=int, default=640)

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--ground-truth", type=Path, required=True)
    eval_parser.add_argument("--predictions", type=Path, required=True)
    eval_parser.add_argument("--experiment-id", required=True)

    val_parser = subparsers.add_parser("validate")
    val_parser.add_argument("--checkpoint", type=Path, required=True)
    val_parser.add_argument("--experiment-id", required=True)
    val_parser.add_argument("--batch", type=int, default=8)
    val_parser.add_argument("--imgsz", type=int, default=640)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "verify":
        payload = validate_locked_inputs()
        write_json(args.output, payload)
    elif args.command == "benchmark":
        payload = benchmark_resources(args.output)
    elif args.command == "train":
        payload = train_baseline(args.config, smoke=args.smoke)
    elif args.command == "export-gt":
        payload = export_coco_ground_truth(args.split, args.output_dir)
    elif args.command == "predict":
        payload = predict_test(
            args.checkpoint,
            args.experiment_id,
            batch=args.batch,
            imgsz=args.imgsz,
        )
    elif args.command == "evaluate":
        payload = evaluate_coco(args.ground_truth, args.predictions, args.experiment_id)
    elif args.command == "validate":
        payload = validate_with_ultralytics(
            args.checkpoint,
            args.experiment_id,
            batch=args.batch,
            imgsz=args.imgsz,
        )
    else:
        raise ValueError(f"unsupported command: {args.command}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "2026")
    main()
