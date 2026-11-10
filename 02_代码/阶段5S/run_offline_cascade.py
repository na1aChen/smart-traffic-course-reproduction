from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
import types
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import sklearn
import torch
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIRST_PROJECT_DEFAULT = Path(
    r"<LOCAL_ROOT>\第一次大作业\代码"
    r"\smart_traffic_itransformer_run_20261022"
)
DETECTION_CSV_DEFAULT = (
    PROJECT_ROOT
    / "03_实验"
    / "诊断"
    / "BASELINE-001"
    / "analysis"
    / "test_pred_events.csv"
)
EVIDENCE_DIR_DEFAULT = PROJECT_ROOT / "01_立项" / "阶段5S_证据"
CHECKPOINT_RELATIVE = (
    Path("checkpoints")
    / "PEMS08_imp_time_pl96_iTransformer_TA_PEMS_sl96_pl96_dm512_el4_"
    "ms0_hr0_th1_tm1_ut1_s2023_ns1_lr0.0005_bs16_Imp_0"
    / "checkpoint.pth"
)
DATASET_RELATIVE = Path("dataset") / "PEMS" / "PEMS08.npz"
MODEL_RELATIVE = Path("model") / "iTransformer_TA.py"

REFERENCE_MSE = 0.193726
REFERENCE_MAE = 0.263401
REFERENCE_TOLERANCE = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="阶段5S：检测计数到 PEMS08 流量预测的离线模块级联"
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "all"),
        default="all",
        help="smoke 只做单窗口联调；all 额外复算完整测试集",
    )
    parser.add_argument(
        "--first-project",
        type=Path,
        default=FIRST_PROJECT_DEFAULT,
        help="第一次大作业预测项目目录",
    )
    parser.add_argument(
        "--detection-csv",
        type=Path,
        default=DETECTION_CSV_DEFAULT,
        help="结构化检测事件表",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EVIDENCE_DIR_DEFAULT,
        help="阶段5S证据输出目录",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="推理设备",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--detector-bins", type=int, default=96)
    parser.add_argument("--replace-channel", type=int, default=169)
    parser.add_argument("--test-window", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label}不存在: {path}")


def install_unused_reformer_shim() -> None:
    """Allow the old shared attention module to import without Reformer.

    The bundled prediction entry imports every model, including Reformer, even
    though the selected iTransformer_TA checkpoint never instantiates LSH
    attention. Installing that unrelated backend would attempt to upgrade the
    working CUDA PyTorch. This shim only permits the import and fails loudly if
    LSH attention is ever requested.
    """

    if "reformer_pytorch" in sys.modules:
        return

    module = types.ModuleType("reformer_pytorch")

    class LSHSelfAttention:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError(
                "阶段5S聚焦推理器不使用 Reformer/LSH；"
                "该占位类被实例化说明执行路径已偏离 iTransformer_TA"
            )

    module.LSHSelfAttention = LSHSelfAttention
    sys.modules["reformer_pytorch"] = module


def load_itransformer_module(first_project: Path) -> tuple[Any, Path]:
    model_path = first_project / MODEL_RELATIVE
    require_file(model_path, "预测模型源码")

    first_project_text = str(first_project)
    if first_project_text not in sys.path:
        sys.path.insert(0, first_project_text)

    install_unused_reformer_shim()
    spec = importlib.util.spec_from_file_location(
        "stage5s_itransformer_ta", model_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法创建模型模块: {model_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, model_path


def make_model_config(seq_len: int, pred_len: int, channels: int) -> SimpleNamespace:
    return SimpleNamespace(
        seq_len=seq_len,
        pred_len=pred_len,
        output_attention=False,
        use_norm=1,
        use_ms=0,
        use_hr=0,
        use_th=1,
        use_tm=1,
        time_dim=4,
        ms_kernels="3,7,15",
        d_model=512,
        d_ff=512,
        e_layers=4,
        n_heads=8,
        dropout=0.1,
        factor=1,
        embed="timeF",
        freq="h",
        activation="gelu",
        enc_in=channels,
    )


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA，但 torch.cuda.is_available() 为 False")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_cyclic_time_marks(
    num_steps: int,
    steps_per_day: int = 288,
    steps_per_week: int = 288 * 7,
) -> np.ndarray:
    time_index = np.arange(num_steps, dtype=np.float64)
    time_of_day = 2.0 * np.pi * (time_index % steps_per_day) / steps_per_day
    day_of_week = 2.0 * np.pi * (time_index % steps_per_week) / steps_per_week
    return np.stack(
        [
            np.sin(time_of_day),
            np.cos(time_of_day),
            np.sin(day_of_week),
            np.cos(day_of_week),
        ],
        axis=1,
    ).astype(np.float32)


def load_pems_data(
    dataset_path: Path,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, dict[str, int]]:
    with np.load(dataset_path, allow_pickle=True) as data:
        if "data" not in data:
            raise KeyError("PEMS08.npz 缺少 data 键")
        raw = np.asarray(data["data"][:, :, 0], dtype=np.float64)

    if raw.ndim != 2:
        raise ValueError(f"PEMS08 第0通道应为二维，实际为 {raw.shape}")
    if not np.isfinite(raw).all():
        # Match the old loader's forward/backward fill semantics conservatively.
        for column in range(raw.shape[1]):
            series = raw[:, column]
            valid = np.flatnonzero(np.isfinite(series))
            if valid.size == 0:
                raise ValueError(f"PEMS08 第{column}列没有有限值")
            first, last = valid[0], valid[-1]
            series[:first] = series[first]
            series[last + 1 :] = series[last]
            for index in range(first + 1, last + 1):
                if not np.isfinite(series[index]):
                    series[index] = series[index - 1]

    num_all = raw.shape[0]
    train_end = int(0.6 * num_all)
    valid_end = int(0.8 * num_all)
    scaler = StandardScaler()
    scaler.fit(raw[:train_end])
    scaled = scaler.transform(raw).astype(np.float32)
    marks = build_cyclic_time_marks(num_all)
    metadata = {
        "num_all_steps": num_all,
        "channels": int(raw.shape[1]),
        "train_end": train_end,
        "valid_end": valid_end,
        "test_steps": int(num_all - valid_end),
    }
    return scaled, marks, scaler, metadata


def image_id_sort_key(image_id: str) -> tuple[int, float | str]:
    try:
        return 0, float(image_id)
    except ValueError:
        return 1, image_id


def aggregate_detection_bins(
    detection_csv: Path,
    num_bins: int,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    require_file(detection_csv, "检测事件表")
    required_columns = {
        "image_id",
        "predicted_class_id",
        "pred_score",
        "pred_box",
    }
    grouped: dict[str, dict[str, float]] = {}
    total_rows = 0

    with detection_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"检测事件表缺少字段: {sorted(missing)}")

        for row in reader:
            total_rows += 1
            try:
                class_id = int(float(row["predicted_class_id"]))
                score = float(row["pred_score"])
                box = json.loads(row["pred_box"])
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if class_id < 0 or not box or not math.isfinite(score):
                continue

            image_id = str(row["image_id"]).strip()
            if not image_id:
                continue
            if image_id not in grouped:
                grouped[image_id] = {"count": 0.0, "score_sum": 0.0}
            grouped[image_id]["count"] += 1.0
            grouped[image_id]["score_sum"] += score

    if not grouped:
        raise ValueError("检测事件表没有可用预测事件")
    if len(grouped) < num_bins:
        raise ValueError(
            f"唯一图像数 {len(grouped)} 小于代理箱数 {num_bins}，无法等频分箱"
        )

    sorted_images = sorted(grouped, key=image_id_sort_key)
    image_to_bin: dict[str, int] = {}
    image_groups: list[list[str]] = [[] for _ in range(num_bins)]
    for rank, image_id in enumerate(sorted_images):
        bin_index = min(rank * num_bins // len(sorted_images), num_bins - 1)
        image_to_bin[image_id] = bin_index
        image_groups[bin_index].append(image_id)

    bin_rows: list[dict[str, Any]] = []
    proxy_raw = np.zeros(num_bins, dtype=np.float64)
    for bin_index, image_ids in enumerate(image_groups):
        detection_count = int(sum(grouped[image_id]["count"] for image_id in image_ids))
        score_sum = float(sum(grouped[image_id]["score_sum"] for image_id in image_ids))
        proxy_raw[bin_index] = score_sum
        bin_rows.append(
            {
                "bin": bin_index,
                "image_id_min": image_ids[0],
                "image_id_max": image_ids[-1],
                "image_count": len(image_ids),
                "detection_count": detection_count,
                "score_sum": score_sum,
            }
        )

    proxy_std = float(proxy_raw.std())
    if proxy_std <= 1e-12:
        raise ValueError("检测代理标准差为 0，无法标准化")
    detector_proxy = ((proxy_raw - proxy_raw.mean()) / proxy_std).astype(
        np.float32
    )
    metadata = {
        "csv_rows": total_rows,
        "valid_prediction_events": int(
            sum(grouped[image_id]["count"] for image_id in grouped)
        ),
        "unique_images": len(grouped),
        "bins": num_bins,
        "proxy_mean_before_standardization": float(proxy_raw.mean()),
        "proxy_std_before_standardization": proxy_std,
        "mapping_type": "image_id_order_proxy",
        "valid_as_time_series": False,
    }
    return bin_rows, detector_proxy, metadata


def write_detector_bins(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "bin",
                "image_id_min",
                "image_id_max",
                "image_count",
                "detection_count",
                "score_sum",
            ],
        )
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["score_sum"] = f"{row['score_sum']:.9f}"
            writer.writerow(output)


def make_batch(
    scaled_data: np.ndarray,
    marks: np.ndarray,
    global_offset: int,
    starts: np.ndarray,
    label_len: int,
    seq_len: int,
    pred_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = []
    targets = []
    input_marks = []
    target_marks = []
    for start in starts:
        start_int = int(start)
        target_start = start_int + seq_len - label_len
        target_end = target_start + label_len + pred_len
        inputs.append(scaled_data[start_int : start_int + seq_len])
        targets.append(scaled_data[target_start:target_end])
        input_marks.append(marks[global_offset + start_int : global_offset + start_int + seq_len])
        target_marks.append(marks[global_offset + target_start : global_offset + target_end])

    x = torch.from_numpy(np.stack(inputs)).float().to(device)
    y = torch.from_numpy(np.stack(targets)).float().to(device)
    x_mark = torch.from_numpy(np.stack(input_marks)).float().to(device)
    y_mark = torch.from_numpy(np.stack(target_marks)).float().to(device)
    return x, y, x_mark, y_mark


def run_model_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    x_mark: torch.Tensor,
    y_mark: torch.Tensor,
    label_len: int,
) -> np.ndarray:
    pred_len = y.shape[1] - label_len
    decoder_input = torch.cat(
        [y[:, :label_len, :], torch.zeros_like(y[:, -pred_len:, :])],
        dim=1,
    )
    output = model(x, x_mark, decoder_input, y_mark)
    return output.detach().cpu().numpy()


def evaluate_full_test(
    model: torch.nn.Module,
    scaled_data: np.ndarray,
    marks: np.ndarray,
    metadata: dict[str, int],
    device: torch.device,
    batch_size: int,
    seq_len: int,
    label_len: int,
    pred_len: int,
) -> dict[str, Any]:
    test_start = metadata["valid_end"]
    test_data = scaled_data[test_start:]
    num_windows = test_data.shape[0] - seq_len - pred_len + 1
    if num_windows <= 0:
        raise ValueError("测试段长度不足以构造一个预测窗口")

    absolute_sum = 0.0
    squared_sum = 0.0
    value_count = 0
    started = time.perf_counter()
    model.eval()

    with torch.inference_mode():
        for offset in range(0, num_windows, batch_size):
            starts = np.arange(offset, min(offset + batch_size, num_windows))
            x, y, x_mark, y_mark = make_batch(
                test_data,
                marks,
                test_start,
                starts,
                label_len,
                seq_len,
                pred_len,
                device,
            )
            prediction = run_model_batch(
                model, x, y, x_mark, y_mark, label_len
            )
            target = y[:, -pred_len:, :].detach().cpu().numpy()
            difference = prediction.astype(np.float64) - target.astype(np.float64)
            absolute_sum += float(np.abs(difference).sum())
            squared_sum += float(np.square(difference).sum())
            value_count += int(difference.size)

    mae = absolute_sum / value_count
    mse = squared_sum / value_count
    return {
        "windows": int(num_windows),
        "values": int(value_count),
        "mse": mse,
        "mae": mae,
        "rmse": math.sqrt(mse),
        "batch_size": batch_size,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "reference_mse": REFERENCE_MSE,
        "reference_mae": REFERENCE_MAE,
        "absolute_mse_difference": abs(mse - REFERENCE_MSE),
        "absolute_mae_difference": abs(mae - REFERENCE_MAE),
        "tolerance": REFERENCE_TOLERANCE,
    }


def predict_single_window(
    model: torch.nn.Module,
    scaled_data: np.ndarray,
    marks: np.ndarray,
    metadata: dict[str, int],
    detector_proxy: np.ndarray,
    replace_channel: int,
    window_index: int,
    device: torch.device,
    seq_len: int,
    label_len: int,
    pred_len: int,
) -> dict[str, Any]:
    test_start = metadata["valid_end"]
    test_data = scaled_data[test_start:]
    max_window = test_data.shape[0] - seq_len - pred_len
    if not 0 <= window_index <= max_window:
        raise ValueError(
            f"test-window 必须在 0..{max_window}，实际为 {window_index}"
        )

    clean_input = test_data[window_index : window_index + seq_len].copy()
    injected_input = clean_input.copy()
    injected_input[:, replace_channel] = detector_proxy

    target_start = window_index + seq_len - label_len
    target_end = target_start + label_len + pred_len
    target_window = test_data[target_start:target_end].copy()

    x_clean = torch.from_numpy(clean_input[None]).float().to(device)
    x_injected = torch.from_numpy(injected_input[None]).float().to(device)
    y = torch.from_numpy(target_window[None]).float().to(device)
    x_mark = torch.from_numpy(
        marks[
            test_start + window_index : test_start + window_index + seq_len
        ][None]
    ).float().to(device)
    y_mark = torch.from_numpy(
        marks[test_start + target_start : test_start + target_end][None]
    ).float().to(device)

    with torch.inference_mode():
        baseline_prediction = run_model_batch(
            model, x_clean, y, x_mark, y_mark, label_len
        )[0]
        injected_prediction = run_model_batch(
            model, x_injected, y, x_mark, y_mark, label_len
        )[0]

    target_prediction = y[:, -pred_len:, :].detach().cpu().numpy()[0]
    difference = injected_prediction.astype(np.float64) - baseline_prediction.astype(
        np.float64
    )
    baseline_mse = float(np.mean((baseline_prediction - target_prediction) ** 2))
    baseline_mae = float(np.mean(np.abs(baseline_prediction - target_prediction)))
    injected_mse = float(np.mean((injected_prediction - target_prediction) ** 2))
    injected_mae = float(np.mean(np.abs(injected_prediction - target_prediction)))
    baseline_scale = float(np.mean(np.abs(baseline_prediction))) + 1e-12
    return {
        "window_index": window_index,
        "replace_channel": replace_channel,
        "baseline_prediction": baseline_prediction,
        "injected_prediction": injected_prediction,
        "target": target_prediction,
        "clean_input": clean_input,
        "injected_input": injected_input,
        "detector_proxy": detector_proxy,
        "summary": {
            "prediction_shape": list(injected_prediction.shape),
            "mean_abs_prediction_delta": float(np.mean(np.abs(difference))),
            "max_abs_prediction_delta": float(np.max(np.abs(difference))),
            "relative_mean_abs_prediction_delta": float(
                np.mean(np.abs(difference)) / baseline_scale
            ),
            "window_baseline_mse": baseline_mse,
            "window_baseline_mae": baseline_mae,
            "window_injected_mse_diagnostic_only": injected_mse,
            "window_injected_mae_diagnostic_only": injected_mae,
            "injected_metrics_are_not_accuracy_evidence": True,
        },
    }


def write_forecast_preview(path: Path, result: dict[str, Any]) -> None:
    baseline_mean = result["baseline_prediction"].mean(axis=1)
    injected_mean = result["injected_prediction"].mean(axis=1)
    target_mean = result["target"].mean(axis=1)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step",
                "detector_proxy",
                "baseline_mean_forecast",
                "injected_mean_forecast",
                "target_mean",
            ]
        )
        for step in range(len(baseline_mean)):
            writer.writerow(
                [
                    step,
                    f"{result['detector_proxy'][step]:.9f}",
                    f"{baseline_mean[step]:.9f}",
                    f"{injected_mean[step]:.9f}",
                    f"{target_mean[step]:.9f}",
                ]
            )


def make_plot(path: Path, result: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.arange(len(result["detector_proxy"]))
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(steps, result["detector_proxy"], color="#c44e52", linewidth=1.8)
    axes[0].set_ylabel("z-score")
    axes[0].set_title("Detector count proxy (image-ID order, not real time)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        steps,
        result["baseline_prediction"].mean(axis=1),
        label="Baseline forecast mean",
        linewidth=1.8,
    )
    axes[1].plot(
        steps,
        result["injected_prediction"].mean(axis=1),
        label="Injected forecast mean",
        linewidth=1.8,
    )
    axes[1].plot(
        steps,
        result["target"].mean(axis=1),
        label="PEMS target mean",
        linewidth=1.2,
        alpha=0.75,
    )
    axes[1].set_xlabel("Forecast step")
    axes[1].set_ylabel("Normalized value")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def build_checks(
    detector_meta: dict[str, Any],
    detector_proxy: np.ndarray,
    cascade_result: dict[str, Any],
    full_metrics: dict[str, Any] | None,
) -> dict[str, bool]:
    checks = {
        "detection_events_readable": detector_meta["valid_prediction_events"] > 0,
        "detector_proxy_length_96": detector_proxy.shape == (96,),
        "detector_proxy_finite": bool(np.isfinite(detector_proxy).all()),
        "pems_input_shape_96x170": cascade_result["injected_input"].shape
        == (96, 170),
        "checkpoint_loaded_strict": bool(cascade_result["checkpoint_loaded_strict"]),
        "prediction_shape_96x170": cascade_result["injected_prediction"].shape
        == (96, 170),
        "prediction_finite": bool(
            np.isfinite(cascade_result["injected_prediction"]).all()
        ),
        "injection_effect_nonzero": cascade_result["summary"][
            "mean_abs_prediction_delta"
        ]
        > 0,
    }
    if full_metrics is not None:
        checks["reference_metrics_reproduced"] = bool(
            full_metrics["absolute_mse_difference"] <= REFERENCE_TOLERANCE
            and full_metrics["absolute_mae_difference"] <= REFERENCE_TOLERANCE
        )
    return checks


def main() -> int:
    args = parse_args()
    first_project = args.first_project.resolve()
    detection_csv = args.detection_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = first_project / DATASET_RELATIVE
    checkpoint_path = first_project / CHECKPOINT_RELATIVE
    require_file(dataset_path, "PEMS08 数据")
    require_file(checkpoint_path, "预测 checkpoint")

    device = choose_device(args.device)
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    model_module, model_path = load_itransformer_module(first_project)
    scaled_data, marks, _, data_metadata = load_pems_data(dataset_path)
    seq_len = 96
    label_len = 48
    pred_len = 96
    channels = data_metadata["channels"]
    if channels != 170:
        raise ValueError(f"本联调固定170通道，实际为 {channels}")

    model_config = make_model_config(seq_len, pred_len, channels)
    model = model_module.Model(model_config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    if not hasattr(state_dict, "keys"):
        raise TypeError("checkpoint 不是可加载的 state_dict")

    load_result = model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    print("checkpoint loaded in strict mode")

    bin_rows, detector_proxy, detector_meta = aggregate_detection_bins(
        detection_csv, args.detector_bins
    )
    if args.detector_bins != seq_len:
        raise ValueError(
            f"当前接口要求检测代理长度等于 seq_len={seq_len}，"
            f"实际 bins={args.detector_bins}"
        )
    write_detector_bins(output_dir / "detector_bins.csv", bin_rows)

    full_metrics = None
    if args.mode == "all":
        print("running full PEMS08 test-set checkpoint verification...")
        full_metrics = evaluate_full_test(
            model,
            scaled_data,
            marks,
            data_metadata,
            device,
            args.batch_size,
            seq_len,
            label_len,
            pred_len,
        )
        print(
            "full test: "
            f"mse={full_metrics['mse']:.9f}, "
            f"mae={full_metrics['mae']:.9f}, "
            f"elapsed={full_metrics['elapsed_seconds']:.1f}s"
        )

    print("running detector-to-forecast cascade example...")
    cascade_result = predict_single_window(
        model,
        scaled_data,
        marks,
        data_metadata,
        detector_proxy,
        args.replace_channel,
        args.test_window,
        device,
        seq_len,
        label_len,
        pred_len,
    )
    cascade_result["checkpoint_loaded_strict"] = True

    preview_path = output_dir / "forecast_preview.csv"
    write_forecast_preview(preview_path, cascade_result)
    np.savez_compressed(
        output_dir / "cascade_predictions.npz",
        baseline_prediction=cascade_result["baseline_prediction"],
        injected_prediction=cascade_result["injected_prediction"],
        target=cascade_result["target"],
        detector_proxy=cascade_result["detector_proxy"],
        clean_input=cascade_result["clean_input"],
        injected_input=cascade_result["injected_input"],
    )

    if not args.no_plot:
        make_plot(output_dir / "cascade_difference.png", cascade_result)

    checks = build_checks(
        detector_meta, detector_proxy, cascade_result, full_metrics
    )
    asset_check = {
        "generated_at": utc_now(),
        "experiment_id": "SYS5S-001",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "pems": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
            "metadata": data_metadata,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "state_entries": len(state_dict),
            "parameter_tensors": int(
                sum(value.numel() for value in state_dict.values())
            ),
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
            "strict_load": not load_result.missing_keys
            and not load_result.unexpected_keys,
        },
        "model_source": {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
            "import_strategy": (
                "focused import of iTransformer_TA; unused Reformer backend "
                "provided an explicit import-only shim"
            ),
        },
        "detection_events": {
            "path": str(detection_csv),
            "sha256": sha256_file(detection_csv),
            **detector_meta,
        },
    }

    forecast_result = {
        "generated_at": utc_now(),
        "experiment_id": "SYS5S-001",
        "chain_type": "offline_module_cascade_prototype",
        "valid_as_real_end_to_end_traffic_validation": False,
        "detector_source": str(detection_csv),
        "forecast_source": str(first_project),
        "detector_adapter": detector_meta,
        "cascade_example": {
            key: value
            for key, value in cascade_result.items()
            if key
            not in {
                "baseline_prediction",
                "injected_prediction",
                "target",
                "clean_input",
                "injected_input",
                "detector_proxy",
            }
        },
        "full_test_checkpoint_verification": full_metrics,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "limitations": [
            "TT100K 图像不是连续时间序列，图像 ID 顺序不能解释为真实时间。",
            "检测对象是交通标志类别，不是车辆流量或 PEMS08 传感器观测。",
            "用检测代理替换第169通道是零重训工程适配器，不是训练好的融合模型。",
            "注入窗口的 MSE/MAE 只作诊断，不能用于声称预测性能提升。",
            "完整测试集指标只用于验证复用 checkpoint 的可加载性和评估口径。",
        ],
    }

    with (output_dir / "阶段5S_资产检查.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(asset_check, handle, ensure_ascii=False, indent=2)
    with (output_dir / "forecast_result.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(forecast_result, handle, ensure_ascii=False, indent=2)

    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print("mean absolute prediction delta:", cascade_result["summary"][
        "mean_abs_prediction_delta"
    ])
    print(f"evidence: {output_dir}")
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
