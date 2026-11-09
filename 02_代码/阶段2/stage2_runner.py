"""Stage 2 runner for the locked YOLOv10-N / TT100K baseline.

The official YOLOv10 source tree is used in place and is not modified. This
runner only supplies project paths and records reproducible evidence.
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import os
import platform
import random
import sys
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SOURCE_ROOT = PROJECT_ROOT / "02_代码" / "源码" / "yolov10"
WEIGHTS_HF_DIR = PROJECT_ROOT / "02_代码" / "权重" / "yolov10n_hf"
WEIGHTS_PT = PROJECT_ROOT / "02_代码" / "权重" / "yolov10n.pt"
DATA_YAML = PROJECT_ROOT / "03_实验" / "datasets" / "TT100K" / "yolo" / "tt100k.yaml"
EXP_ROOT = PROJECT_ROOT / "03_实验" / "baseline_yolov10n"
EVIDENCE_ROOT = PROJECT_ROOT / "01_立项" / "阶段2_证据"

sys.path.insert(0, str(SOURCE_ROOT))

import torch
import yaml
from ultralytics import YOLOv10
from ultralytics.utils import patches as ultralytics_patches
from ultralytics.utils.torch_utils import get_flops


def install_unicode_path_save_patch() -> None:
    """Avoid PyTorch 2.0.1's Windows save failure for non-ASCII paths."""
    original_save = ultralytics_patches._torch_save

    def unicode_safe_save(obj, target, *args, **kwargs):
        if isinstance(target, (str, Path)) and not str(target).isascii():
            buffer = io.BytesIO()
            original_save(obj, buffer, *args, **kwargs)
            Path(target).write_bytes(buffer.getvalue())
            return None
        return original_save(obj, target, *args, **kwargs)

    ultralytics_patches._torch_save = unicode_safe_save


install_unicode_path_save_patch()


class Tee:
    """Write text to multiple streams."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(
            getattr(stream, "isatty", lambda: False)() for stream in self.streams
        )


@contextmanager
def tee_console(log_path: Path) -> Iterator[None]:
    """Persist console output while retaining live terminal output."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with log_path.open("w", encoding="utf-8") as handle:
        stream = Tee(original_stdout, handle)
        sys.stdout = stream
        sys.stderr = stream
        try:
            yield
        finally:
            stream.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")  # noqa: UP017


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary_path.replace(path)


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def verify_weight_equivalence() -> dict[str, Any]:
    """Prove that the trainable .pt carrier equals the locked HF weights.

    Ultralytics 8.1.34 reconstructs the trainer model and only carries weights
    forward when the wrapper has a non-empty checkpoint dictionary. The HF
    ``from_pretrained`` path has ``ckpt=None``, so directly calling ``train``
    after it would silently start without the locked weights. We therefore use
    the local .pt carrier after proving every tensor is identical.
    """
    if not WEIGHTS_HF_DIR.is_dir():
        raise FileNotFoundError(
            f"Locked HF weight directory not found: {WEIGHTS_HF_DIR}"
        )
    if not WEIGHTS_PT.is_file():
        raise FileNotFoundError(f"Trainable weight carrier not found: {WEIGHTS_PT}")

    hf_model = YOLOv10.from_pretrained(str(WEIGHTS_HF_DIR), local_files_only=True)
    pt_model = YOLOv10(str(WEIGHTS_PT))
    hf_state = hf_model.model.state_dict()
    pt_state = pt_model.model.state_dict()
    keys_equal = list(hf_state) == list(pt_state)
    mismatches: list[str] = []
    if keys_equal:
        for key in hf_state:
            if (
                hf_state[key].shape != pt_state[key].shape
                or hf_state[key].dtype != pt_state[key].dtype
                or not torch.equal(hf_state[key], pt_state[key])
            ):
                mismatches.append(key)
    else:
        mismatches = sorted(set(hf_state).symmetric_difference(pt_state))

    payload = {
        "created_at": utc_now(),
        "hf_directory": str(WEIGHTS_HF_DIR),
        "pt_carrier": str(WEIGHTS_PT),
        "hf_ckpt_is_empty": not bool(hf_model.ckpt),
        "pt_ckpt_is_nonempty": bool(pt_model.ckpt),
        "tensor_count_hf": len(hf_state),
        "tensor_count_pt": len(pt_state),
        "keys_equal": keys_equal,
        "all_tensors_exactly_equal": keys_equal and not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "training_initialization_rule": (
            "Use YOLOv10(local .pt) so Model.train carries the non-empty checkpoint into "
            "the reconstructed 45-class model; HF remains the provenance authority."
        ),
    }
    write_json(EVIDENCE_ROOT / "权重等价性校验.json", payload)
    if not payload["all_tensors_exactly_equal"]:
        raise RuntimeError(
            f"HF weights and .pt carrier differ: {len(mismatches)} tensors"
        )
    if (
        payload["hf_ckpt_is_empty"] is not True
        or payload["pt_ckpt_is_nonempty"] is not True
    ):
        raise RuntimeError("Unexpected checkpoint-wrapper behavior; refusing to train")
    return payload


def load_trainable_model() -> YOLOv10:
    """Load the verified .pt carrier that actually transfers pretrained weights."""
    if not WEIGHTS_PT.is_file():
        raise FileNotFoundError(f"Trainable weight carrier not found: {WEIGHTS_PT}")
    return YOLOv10(str(WEIGHTS_PT))


def register_memory_callbacks(model: YOLOv10) -> dict[str, Any]:
    """Track actual batch allocation peaks during training."""
    state: dict[str, Any] = {
        "max_memory_allocated_bytes": 0,
        "max_memory_reserved_bytes": 0,
        "train_batches": 0,
        "train_epoch_seconds": None,
        "train_dataset_images": None,
    }

    def on_train_start(trainer) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            state["max_memory_allocated_bytes"] = torch.cuda.memory_allocated()
            state["max_memory_reserved_bytes"] = torch.cuda.memory_reserved()
        state["train_started_perf_counter"] = time.perf_counter()
        state["train_dataset_images"] = len(trainer.train_loader.dataset)

    def on_train_batch_end(trainer) -> None:
        state["train_batches"] += 1
        if torch.cuda.is_available():
            state["max_memory_allocated_bytes"] = max(
                state["max_memory_allocated_bytes"],
                torch.cuda.max_memory_allocated(),
            )
            state["max_memory_reserved_bytes"] = max(
                state["max_memory_reserved_bytes"],
                torch.cuda.max_memory_reserved(),
            )

    def on_train_epoch_end(trainer) -> None:
        if state["train_epoch_seconds"] is None:
            state["train_epoch_seconds"] = (
                time.perf_counter() - state["train_started_perf_counter"]
            )
            if state["train_epoch_seconds"] > 0 and state["train_dataset_images"]:
                state["train_images_per_second"] = (
                    state["train_dataset_images"] / state["train_epoch_seconds"]
                )

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    return state


def model_metadata(model: YOLOv10, imgsz: int) -> dict[str, Any]:
    parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
    try:
        flops = float(get_flops(model.model, imgsz=imgsz))
        if flops <= 0:
            raise ValueError(f"get_flops returned non-positive value: {flops}")
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - environment-specific diagnostic
        flops = None
        flops_error = f"{type(exc).__name__}: {exc}"
    else:
        flops_error = None
    return {
        "parameters": parameter_count,
        "flops_gflops_at_imgsz": flops,
        "flops_error": flops_error,
    }


def run_probe(args: argparse.Namespace) -> int:
    """Run short, non-final training probes for memory and throughput."""
    probe_root = EXP_ROOT / "probes"
    summary_path = EVIDENCE_ROOT / "stage2_probe_summary.json"
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    configurations = []
    for item in args.matrix.split(","):
        imgsz_text, batch_text = item.strip().lower().split("x", maxsplit=1)
        configurations.append((int(imgsz_text), int(batch_text)))
    if not configurations:
        raise ValueError("Probe matrix is empty")
    results: list[dict[str, Any]] = []

    for imgsz, batch in configurations:
        run_name = f"probe_{run_stamp}_i{imgsz}_b{batch}"
        run_dir = probe_root / run_name
        log_path = run_dir / "console.log"
        result: dict[str, Any] = {
            "run_name": run_name,
            "imgsz": imgsz,
            "batch": batch,
            "fraction": args.fraction,
            "seed": args.seed,
            "started_at": utc_now(),
            "status": "started",
        }
        started = time.perf_counter()
        try:
            set_reproducibility(args.seed)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = load_trainable_model()
            peaks = register_memory_callbacks(model)
            run_dir.mkdir(parents=True, exist_ok=False)
            (run_dir / "weights").mkdir(parents=True, exist_ok=True)
            with tee_console(log_path):
                model.train(
                    data=str(DATA_YAML),
                    epochs=1,
                    fraction=args.fraction,
                    batch=batch,
                    imgsz=imgsz,
                    device=args.device,
                    workers=args.workers,
                    project=str(probe_root),
                    name=run_name,
                    exist_ok=True,
                    seed=args.seed,
                    deterministic=True,
                    amp=True,
                    val=False,
                    plots=False,
                    save=True,
                    cache=False,
                    verbose=True,
                )
            result.update(peaks)
            result["status"] = "passed"
        except torch.cuda.OutOfMemoryError as exc:
            result["status"] = "oom"
            result["error"] = f"{type(exc).__name__}: {exc}"
            if not log_path.exists():
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(result["error"] + "\n", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            result["status"] = "failed"
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["traceback"] = traceback.format_exc()
            if not log_path.exists():
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(result["traceback"], encoding="utf-8")
        finally:
            result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            result["log_path"] = str(log_path)
            results.append(result)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_json(
        summary_path,
        {
            "created_at": utc_now(),
            "purpose": "Stage 2 smoke and resource probe; not a final model result",
            "data_yaml": str(DATA_YAML),
            "locked_weights_hf": str(WEIGHTS_HF_DIR),
            "trainable_weight_carrier": str(WEIGHTS_PT),
            "python": sys.version,
            "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "results": results,
        },
    )
    print(f"Probe summary: {summary_path}")
    return 0 if any(item.get("status") == "passed" for item in results) else 1


def run_training(args: argparse.Namespace) -> int:
    """Run the one locked Stage 2 baseline training recipe."""
    run_dir = EXP_ROOT / "train"
    log_path = run_dir / "console.log"
    summary_path = EVIDENCE_ROOT / "stage2_training_summary.json"
    started = time.perf_counter()
    summary: dict[str, Any] = {
        "experiment_id": "BASE_YOLOV10N_TT100K45_S2026",
        "started_at": utc_now(),
        "status": "started",
        "data_yaml": str(DATA_YAML),
        "weights_hf": str(WEIGHTS_HF_DIR),
        "trainable_weight_carrier": str(WEIGHTS_PT),
        "command_arguments": {
            key: value for key, value in vars(args).items() if key != "func"
        },
    }
    try:
        set_reproducibility(args.seed)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        weight_check = verify_weight_equivalence()
        model = load_trainable_model()
        summary["weight_equivalence"] = weight_check
        summary["model"] = model_metadata(model, imgsz=args.imgsz)
        peaks = register_memory_callbacks(model)
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "weights").mkdir(parents=True, exist_ok=True)
        with tee_console(log_path):
            model.train(
                data=str(DATA_YAML),
                epochs=args.epochs,
                batch=args.batch,
                imgsz=args.imgsz,
                device=args.device,
                workers=args.workers,
                optimizer=args.optimizer,
                project=str(EXP_ROOT),
                name="train",
                exist_ok=True,
                seed=args.seed,
                deterministic=True,
                amp=True,
                patience=args.patience,
                close_mosaic=args.close_mosaic,
                plots=True,
                save=True,
                save_period=args.save_period,
                val=True,
                cache=False,
                verbose=True,
            )
        summary.update(peaks)
        summary["status"] = "passed"
        summary["best_checkpoint"] = str(run_dir / "weights" / "best.pt")
        summary["last_checkpoint"] = str(run_dir / "weights" / "last.pt")
    except Exception as exc:  # noqa: BLE001
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()
        if not log_path.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(summary["traceback"], encoding="utf-8")
    finally:
        summary["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        summary["finished_at"] = utc_now()
        write_json(summary_path, summary)
    print(f"Training summary: {summary_path}")
    return 0 if summary["status"] == "passed" else 1


def _plain_metrics(
    metrics: Any, imgsz: int
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    results_dict = {
        str(key): float(value) for key, value in metrics.results_dict.items()
    }
    per_class: list[dict[str, Any]] = []
    names = metrics.names
    for metric_index, class_id in enumerate(metrics.ap_class_index):
        precision, recall, map50, map5095 = metrics.class_result(metric_index)
        class_key = str(class_id)
        class_id_int = int(class_id)
        if isinstance(names, dict):
            class_name = names.get(
                class_id_int,
                names.get(class_key, str(class_id_int)),
            )
        elif isinstance(names, list):
            class_name = names[class_id_int]
        else:
            class_name = str(class_id_int)
        per_class.append(
            {
                "class_id": class_id_int,
                "class_name": class_name,
                "precision": float(precision),
                "recall": float(recall),
                "map50": float(map50),
                "map50_95": float(map5095),
            }
        )
    results_dict["inference_ms_per_image"] = float(metrics.speed.get("inference", 0.0))
    results_dict["preprocess_ms_per_image"] = float(
        metrics.speed.get("preprocess", 0.0)
    )
    results_dict["postprocess_ms_per_image"] = float(
        metrics.speed.get("postprocess", 0.0)
    )
    results_dict["imgsz"] = float(imgsz)
    return results_dict, per_class


def _write_per_class_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["class_id", "class_name", "precision", "recall", "map50", "map50_95"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_coco_ground_truth(
    data_yaml: Path,
    split: str,
    output_path: Path,
    predictions_dir: Path,
) -> tuple[Path, Path, Path]:
    """Convert locked labels and saved YOLO txt predictions to COCO format."""
    from PIL import Image

    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    data_root = Path(data["path"])
    image_dir = data_root / data[split]
    label_dir = data_root / "labels" / split
    names = data["names"]

    categories = [
        {"id": index + 1, "name": str(name), "supercategory": "traffic_sign"}
        for index, name in enumerate(names)
    ]
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1

    for image_path in sorted(image_dir.glob("*.jpg")):
        image_id = (
            int(image_path.stem) if image_path.stem.isdigit() else image_path.stem
        )
        with Image.open(image_path) as image:
            width, height = image.size
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            class_id, cx, cy, box_width, box_height = map(float, line.split()[:5])
            x = (cx - box_width / 2.0) * width
            y = (cy - box_height / 2.0) * height
            absolute_width = box_width * width
            absolute_height = box_height * height
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(class_id) + 1,
                    "bbox": [x, y, absolute_width, absolute_height],
                    "area": absolute_width * absolute_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        output_path,
        {
            "info": {
                "description": "TT100K 45-class locked split converted from YOLO labels"
            },
            "images": images,
            "annotations": annotations,
            "categories": categories,
        },
    )

    image_sizes = {item["id"]: (item["width"], item["height"]) for item in images}
    raw_predictions: list[dict[str, Any]] = []
    converted_predictions: list[dict[str, Any]] = []
    if not predictions_dir.is_dir():
        raise FileNotFoundError(
            f"Raw YOLO prediction directory not found: {predictions_dir}"
        )
    for image_path in sorted(image_dir.glob("*.jpg")):
        image_id = (
            int(image_path.stem) if image_path.stem.isdigit() else image_path.stem
        )
        width, height = image_sizes[image_id]
        prediction_path = predictions_dir / f"{image_path.stem}.txt"
        if not prediction_path.is_file():
            continue
        for line in prediction_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            values = line.split()
            if len(values) < 6:
                raise ValueError(
                    f"Expected class/cx/cy/w/h/conf in {prediction_path}: {line}"
                )
            class_id = int(float(values[0]))
            cx, cy, box_width, box_height, score = map(float, values[1:6])
            x = (cx - box_width / 2.0) * width
            y = (cy - box_height / 2.0) * height
            absolute_width = box_width * width
            absolute_height = box_height * height
            raw_item = {
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [
                    round(x, 3),
                    round(y, 3),
                    round(absolute_width, 3),
                    round(absolute_height, 3),
                ],
                "score": round(score, 5),
            }
            raw_predictions.append(raw_item)
            converted_predictions.append({**raw_item, "category_id": class_id + 1})

    if not raw_predictions:
        raise RuntimeError(f"No raw predictions found in {predictions_dir}")
    raw_path = output_path.with_name("predictions_category0.json")
    converted_path = output_path.with_name("predictions_category1.json")
    write_json(raw_path, raw_predictions)
    write_json(converted_path, converted_predictions)
    return output_path, raw_path, converted_path


def recompute_metrics(
    data_yaml: Path,
    split: str,
    eval_dir: Path,
    predictions_dir: Path,
) -> dict[str, Any]:
    """Recompute COCO bbox mAP from saved raw predictions and labels."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt_path = eval_dir / "ground_truth_coco.json"
    gt_path, prediction_path, converted_prediction_path = _build_coco_ground_truth(
        data_yaml,
        split,
        gt_path,
        predictions_dir,
    )
    ground_truth = COCO(str(gt_path))
    predictions = ground_truth.loadRes(str(converted_prediction_path))
    evaluator = COCOeval(ground_truth, predictions, "bbox")
    evaluator.params.imgIds = sorted(ground_truth.getImgIds())
    evaluator.params.catIds = sorted(ground_truth.getCatIds())
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    stats = evaluator.stats
    return {
        "ground_truth": str(gt_path),
        "predictions_category0": str(prediction_path),
        "predictions_category1": str(converted_prediction_path),
        "map50_95": float(stats[0]),
        "map50": float(stats[1]),
        "map75": float(stats[2]),
        "map_small": float(stats[3]),
        "map_medium": float(stats[4]),
        "map_large": float(stats[5]),
        "mar100": float(stats[8]),
        "images": len(ground_truth.getImgIds()),
        "categories": len(ground_truth.getCatIds()),
    }


def run_evaluation(args: argparse.Namespace) -> int:
    """Evaluate the selected checkpoint once on the untouched test split."""
    checkpoint = EXP_ROOT / "train" / "weights" / "best.pt"
    eval_dir = EXP_ROOT / "test_eval"
    log_path = eval_dir / "console.log"
    summary_path = EVIDENCE_ROOT / "stage2_test_evaluation.json"
    started = time.perf_counter()
    summary: dict[str, Any] = {
        "experiment_id": "BASE_YOLOV10N_TT100K45_S2026",
        "started_at": utc_now(),
        "status": "started",
        "checkpoint": str(checkpoint),
        "data_yaml": str(DATA_YAML),
        "split": "test",
        "selection_rule": "best.pt selected by validation fitness; test evaluated once",
    }
    try:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Best checkpoint not found: {checkpoint}")
        set_reproducibility(args.seed)
        model = YOLOv10(str(checkpoint))
        summary["model"] = model_metadata(model, imgsz=args.imgsz)
        eval_dir.mkdir(parents=True, exist_ok=False)
        with tee_console(log_path):
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
                # Match the standard COCO evaluator's maxDets=100 so the YOLO
                # summary and the independently recomputed metric use the same
                # detection cap.
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
            DATA_YAML,
            "test",
            eval_dir,
            eval_dir / "labels",
        )
        summary["raw_predictions"] = str(eval_dir / "predictions_category0.json")
        summary["status"] = "passed"
    except Exception as exc:  # noqa: BLE001
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()
        if not log_path.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(summary["traceback"], encoding="utf-8")
    finally:
        summary["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        summary["finished_at"] = utc_now()
        write_json(summary_path, summary)
    print(f"Evaluation summary: {summary_path}")
    return 0 if summary["status"] == "passed" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 2 YOLOv10-N TT100K runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", default="0")
    common.add_argument("--workers", type=int, default=4)
    common.add_argument("--seed", type=int, default=2026)

    probe = subparsers.add_parser("probe", parents=[common])
    probe.add_argument("--fraction", type=float, default=0.04)
    probe.add_argument("--matrix", default="512x4,640x4,512x8,640x8")
    probe.set_defaults(func=run_probe)

    train = subparsers.add_parser("train", parents=[common])
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--optimizer", default="auto")
    train.add_argument("--patience", type=int, default=20)
    train.add_argument("--close-mosaic", type=int, default=10)
    train.add_argument("--save-period", type=int, default=10)
    train.set_defaults(func=run_training)

    evaluate = subparsers.add_parser("evaluate", parents=[common])
    evaluate.add_argument("--batch", type=int, default=8)
    evaluate.add_argument("--imgsz", type=int, default=640)
    evaluate.set_defaults(func=run_evaluation)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    raise SystemExit(parsed.func(parsed))
