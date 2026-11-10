from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_TABLE = PROJECT_ROOT / "03_实验" / "实验总表.csv"
STAGE5_METRICS = (
    PROJECT_ROOT / "01_立项" / "阶段5_证据" / "stage5_experiment_metrics.csv"
)
STAGE5_COMPARISON = (
    PROJECT_ROOT / "01_立项" / "阶段5_证据" / "stage5_comparison.json"
)
ANALYSIS_SUMMARY = (
    PROJECT_ROOT
    / "03_实验"
    / "诊断"
    / "BASELINE-001"
    / "analysis"
    / "analysis_summary.json"
)
FORECAST_RESULT = (
    PROJECT_ROOT / "01_立项" / "阶段5S_证据" / "forecast_result.json"
)
CASCADE_PREDICTIONS = (
    PROJECT_ROOT / "01_立项" / "阶段5S_证据" / "cascade_predictions.npz"
)
EVIDENCE_ROOT = PROJECT_ROOT / "01_立项" / "阶段6_证据"
FIGURE_ROOT = EVIDENCE_ROOT / "主图"
FINAL_TABLE_CSV = EVIDENCE_ROOT / "最终主表.csv"
FINAL_TABLE_MD = EVIDENCE_ROOT / "最终主表.md"
EXPERIMENT_INDEX = EVIDENCE_ROOT / "实验证据索引.csv"
LOSS_REGISTRY = EVIDENCE_ROOT / "损失口径登记.csv"
PERCENT_REGISTRY = EVIDENCE_ROOT / "百分比口径登记.csv"
FIGURE_MANIFEST = EVIDENCE_ROOT / "主图冻结清单.csv"
CHANGE_REGISTER = EVIDENCE_ROOT / "数字变更登记.csv"
HASH_MANIFEST = EVIDENCE_ROOT / "证据SHA256清单.csv"
FREEZE_METADATA = EVIDENCE_ROOT / "冻结元数据.json"
VERSION_RULES = EVIDENCE_ROOT / "数字变更与版本规则.md"
STAGE6_SUMMARY = PROJECT_ROOT / "01_立项" / "阶段总结" / "阶段6_总结.md"
STAGE6_AUDIT = EVIDENCE_ROOT / "阶段6审计.json"

STAGE5S_COMMIT = "f8b736b"
STAGE5S_TAG = "stage-5s-complete"
FREEZE_VERSION = "stage6-evidence-v1.0.0"
MAIN_MODEL = "BASELINE-001"

TT100K_RAW = (
    PROJECT_ROOT / "03_实验" / "datasets" / "TT100K" / "raw" / "data.zip"
)
PEMS08 = Path(
    r"<LOCAL_ROOT>\第一次大作业\代码"
    r"\smart_traffic_itransformer_run_20261022\dataset\PEMS\PEMS08.npz"
)
FLOW_CHECKPOINT = Path(
    r"<LOCAL_ROOT>\第一次大作业\代码"
    r"\smart_traffic_itransformer_run_20261022\checkpoints"
    r"\PEMS08_imp_time_pl96_iTransformer_TA_PEMS_sl96_pl96_dm512_el4_ms0_hr0"
    r"_th1_tm1_ut1_s2023_ns1_lr0.0005_bs16_Imp_0\checkpoint.pth"
)


EXPERIMENTS: list[dict[str, Any]] = [
    {
        "registry_id": "BASE_YOLOV10N_TT100K45_S2026",
        "report_id": "BASELINE-001",
        "stage": "2",
        "role": "最终主模型/主结果",
        "group": "baseline",
        "config": "03_实验/configs/BASELINE-001.yaml",
        "log": "03_实验/logs/BASELINE-001/train_stdout.log",
        "training_summary": "01_立项/阶段2_证据/stage2_training_summary.json",
        "validation_summary": "01_立项/阶段3_证据/stage3_baseline_val_coco_metrics.json",
        "test_summary": "01_立项/阶段2_证据/stage2_test_evaluation.json",
        "results_csv": "03_实验/baseline_yolov10n/train/results.csv",
        "checkpoint": "03_实验/baseline_yolov10n/train/weights/best.pt",
        "predictions": "03_实验/baseline_yolov10n/test_eval/predictions_category0.json",
        "prediction_support": "03_实验/baseline_yolov10n/test_eval/predictions_category1.json",
        "prediction_labels": "03_实验/baseline_yolov10n/test_eval/labels",
        "test_log": "03_实验/logs/BASELINE-001/postprocess.log",
        "notes": "登记表 ID 与证据目录别名不同；两者均保留。",
    },
    {
        "registry_id": "P2-001",
        "report_id": "P2-001",
        "stage": "4",
        "role": "P2负结果/权衡实验",
        "group": "p2",
        "config": "03_实验/configs/P2-001.yaml",
        "log": "03_实验/logs/P2-001/resume_stdout.log",
        "training_summary": "01_立项/阶段4_证据/stage4_training_summary.json",
        "validation_summary": "03_实验/P2-001/val_eval/stage4_val_manifest.json",
        "test_summary": "01_立项/阶段4_证据/stage4_test_evaluation.json",
        "results_csv": "03_实验/P2-001/train/results.csv",
        "checkpoint": "03_实验/P2-001/train/weights/best.pt",
        "predictions": "03_实验/P2-001/test_eval/predictions_category0.json",
        "prediction_support": "03_实验/P2-001/test_eval/predictions_category1.json",
        "prediction_labels": "03_实验/P2-001/test_eval/labels",
        "test_log": "03_实验/logs/P2-001/test_stdout.log",
        "notes": "从第28轮断电恢复并完成40轮。",
    },
    {
        "registry_id": "BASELINE-002-S2027",
        "report_id": "BASELINE-002-S2027",
        "stage": "5",
        "role": "基线第二 seed",
        "group": "baseline",
        "config": "03_实验/configs/BASELINE-002.yaml",
        "log": "03_实验/logs/STAGE5/BASELINE-002-S2027_train.log",
        "training_summary": "01_立项/阶段5_证据/training/BASELINE-002-S2027.json",
        "validation_summary": "01_立项/阶段5_证据/validation/BASELINE-002-S2027.json",
        "test_summary": "01_立项/阶段5_证据/evaluation/BASELINE-002-S2027.json",
        "results_csv": "03_实验/robustness/BASELINE-002-S2027/train/results.csv",
        "checkpoint": "03_实验/robustness/BASELINE-002-S2027/train/weights/best.pt",
        "predictions": (
            "03_实验/robustness/BASELINE-002-S2027/test_eval/"
            "predictions_category0.json"
        ),
        "prediction_support": (
            "03_实验/robustness/BASELINE-002-S2027/test_eval/"
            "predictions_category1.json"
        ),
        "prediction_labels": (
            "03_实验/robustness/BASELINE-002-S2027/test_eval/labels"
        ),
        "test_log": "03_实验/logs/STAGE5/BASELINE-002-S2027_test.log",
        "notes": "用于基线两 seed 均值与标准差。",
    },
    {
        "registry_id": "P2-002-S2027",
        "report_id": "P2-002-S2027",
        "stage": "5",
        "role": "P2第二 seed",
        "group": "p2",
        "config": "03_实验/configs/P2-002.yaml",
        "log": "03_实验/logs/STAGE5/P2-002-S2027_resume.log",
        "training_summary": "01_立项/阶段5_证据/training/P2-002-S2027.json",
        "validation_summary": "01_立项/阶段5_证据/validation/P2-002-S2027.json",
        "test_summary": "01_立项/阶段5_证据/evaluation/P2-002-S2027.json",
        "results_csv": "03_实验/robustness/P2-002-S2027/train/results.csv",
        "checkpoint": "03_实验/robustness/P2-002-S2027/train/weights/best.pt",
        "predictions": (
            "03_实验/robustness/P2-002-S2027/test_eval/"
            "predictions_category0.json"
        ),
        "prediction_support": (
            "03_实验/robustness/P2-002-S2027/test_eval/"
            "predictions_category1.json"
        ),
        "prediction_labels": (
            "03_实验/robustness/P2-002-S2027/test_eval/labels"
        ),
        "test_log": "03_实验/logs/STAGE5/P2-002-S2027_test.log",
        "notes": "第27轮后手动暂停，并从第28轮恢复。",
    },
    {
        "registry_id": "BASELINE-BUDGET48-S2026",
        "report_id": "BASELINE-BUDGET48-S2026",
        "stage": "5",
        "role": "48轮训练预算对照",
        "group": "budget",
        "config": "03_实验/configs/BASELINE-BUDGET48.yaml",
        "log": "03_实验/logs/STAGE5/BASELINE-BUDGET48-S2026_train.log",
        "training_summary": (
            "01_立项/阶段5_证据/training/BASELINE-BUDGET48-S2026.json"
        ),
        "validation_summary": (
            "01_立项/阶段5_证据/validation/BASELINE-BUDGET48-S2026.json"
        ),
        "test_summary": (
            "01_立项/阶段5_证据/evaluation/BASELINE-BUDGET48-S2026.json"
        ),
        "results_csv": (
            "03_实验/robustness/BASELINE-BUDGET48-S2026/train/results.csv"
        ),
        "checkpoint": (
            "03_实验/robustness/BASELINE-BUDGET48-S2026/train/weights/best.pt"
        ),
        "predictions": (
            "03_实验/robustness/BASELINE-BUDGET48-S2026/test_eval/"
            "predictions_category0.json"
        ),
        "prediction_support": (
            "03_实验/robustness/BASELINE-BUDGET48-S2026/test_eval/"
            "predictions_category1.json"
        ),
        "prediction_labels": (
            "03_实验/robustness/BASELINE-BUDGET48-S2026/test_eval/labels"
        ),
        "test_log": "03_实验/logs/STAGE5/BASELINE-BUDGET48-S2026_test.log",
        "notes": "只用于训练预算解释，不用于替换最终主模型。",
    },
    {
        "registry_id": "SYS5S-001",
        "report_id": "SYS5S-001",
        "stage": "5S",
        "role": "离线模块级联原型",
        "group": "prototype",
        "config": "01_立项/阶段5S_证据/阶段5S预注册与接口设计.md",
        "log": "03_实验/logs/STAGE5S/SYS5S-001_run.log",
        "training_summary": None,
        "validation_summary": None,
        "test_summary": "01_立项/阶段5S_证据/forecast_result.json",
        "results_csv": None,
        "checkpoint": (
            "03_实验/baseline_yolov10n/train/weights/best.pt;"
            + str(FLOW_CHECKPOINT)
        ),
        "predictions": "01_立项/阶段5S_证据/cascade_predictions.npz",
        "prediction_support": None,
        "prediction_labels": None,
        "test_log": "03_实验/logs/STAGE5S/SYS5S-001_run.log",
        "notes": "零重训；检测主指标与流量 MSE/MAE 分域记录，跨域不可比较。",
    },
]


DETECTION_IDS = [item["report_id"] for item in EXPERIMENTS[:-1]]
LOSS_COLUMNS = [
    "train/box_om",
    "train/cls_om",
    "train/dfl_om",
    "train/box_oo",
    "train/cls_oo",
    "train/dfl_oo",
    "val/box_om",
    "val/cls_om",
    "val/dfl_om",
    "val/box_oo",
    "val/cls_oo",
    "val/dfl_oo",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def as_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def rel_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON 顶层不是对象: {path}")
    return payload


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="\n") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def sha256_file(path: Path, show_progress: bool = True) -> str:
    size = path.stat().st_size
    if show_progress and size >= 1_000_000_000:
        print(f"计算大文件 SHA-256: {rel_path(path)} ({size / 2**30:.2f} GiB)")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_directory(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    total_bytes = 0
    for item in files:
        relative = item.relative_to(path).as_posix()
        file_hash = sha256_file(item, show_progress=False)
        size = item.stat().st_size
        total_bytes += size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest().upper(), len(files), total_bytes


def f6(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.6f}"


def build_metric_map() -> dict[str, dict[str, str]]:
    return {
        row["experiment_id"]: row
        for row in load_csv(STAGE5_METRICS)
    }


def build_table_map() -> dict[str, dict[str, str]]:
    return {
        row["experiment_id"]: row
        for row in load_csv(EXPERIMENT_TABLE)
    }


def build_experiment_index() -> None:
    metric_rows = build_metric_map()
    table_rows = build_table_map()
    rows: list[dict[str, Any]] = []
    expected_registry_ids = {item["registry_id"] for item in EXPERIMENTS}
    if set(table_rows) != expected_registry_ids:
        raise RuntimeError(
            "实验总表 ID 与阶段6映射不一致: "
            f"table={sorted(table_rows)}, expected={sorted(expected_registry_ids)}"
        )

    for item in EXPERIMENTS:
        report_id = item["report_id"]
        table_row = table_rows[item["registry_id"]]
        metric_row = metric_rows.get(report_id)
        metric_path = item["test_summary"]
        normalized = {
            "registry_experiment_id": item["registry_id"],
            "canonical_experiment_id": report_id,
            "stage": item["stage"],
            "role": item["role"],
            "config_path": item["config"],
            "config_exists": as_path(item["config"]).is_file(),
            "log_path": table_row["log_path"],
            "log_exists": as_path(table_row["log_path"]).is_file(),
            "training_summary_path": item["training_summary"] or "",
            "training_summary_exists": (
                as_path(item["training_summary"]).is_file()
                if item["training_summary"]
                else ""
            ),
            "validation_metrics_path": item["validation_summary"] or "",
            "validation_metrics_exists": (
                as_path(item["validation_summary"]).is_file()
                if item["validation_summary"]
                else ""
            ),
            "test_metrics_path": metric_path,
            "test_metrics_exists": as_path(metric_path).is_file(),
            "checkpoint_path": item["checkpoint"],
            "checkpoint_paths_exist": all(
                as_path(path).is_file() for path in item["checkpoint"].split(";")
            ),
            "primary_prediction_path": item["predictions"],
            "primary_prediction_exists": as_path(item["predictions"]).is_file(),
            "supporting_prediction_path": item["prediction_support"] or "",
            "supporting_prediction_exists": (
                as_path(item["prediction_support"]).is_file()
                if item["prediction_support"]
                else ""
            ),
            "prediction_labels_path": item["prediction_labels"] or "",
            "prediction_labels_exists": (
                as_path(item["prediction_labels"]).is_dir()
                if item["prediction_labels"]
                else ""
            ),
            "test_log_path": item["test_log"],
            "test_log_exists": as_path(item["test_log"]).is_file(),
            "test_metric_scope": (
                "flow_checkpoint_full_test_verification"
                if report_id == "SYS5S-001"
                else "detector_test_coco"
            ),
            "canonical_metrics_source": (
                "阶段5实验指标CSV" if metric_row else "阶段5S预测结果JSON"
            ),
            "notes": item["notes"],
        }
        rows.append(normalized)

    fields = list(rows[0])
    write_csv(EXPERIMENT_INDEX, rows, fields)


def read_precision_recall(item: dict[str, Any]) -> tuple[str, str]:
    if item["report_id"] == "SYS5S-001":
        return "", ""
    payload = load_json(as_path(item["test_summary"]))
    metrics = payload.get("metrics_yolov10", {})
    return (
        f6(metrics.get("metrics/precision(B)")),
        f6(metrics.get("metrics/recall(B)")),
    )


def build_final_table() -> list[dict[str, str]]:
    stage5_metrics = build_metric_map()
    table_rows = build_table_map()
    forecast = load_json(FORECAST_RESULT)
    rows: list[dict[str, str]] = []

    for item in EXPERIMENTS:
        report_id = item["report_id"]
        table_row = table_rows[item["registry_id"]]
        precision, recall = read_precision_recall(item)
        base = {
            "registry_experiment_id": item["registry_id"],
            "experiment_id": report_id,
            "role": item["role"],
            "comparable_group": item["group"],
            "stage": item["stage"],
            "seed": table_row.get("seed", ""),
            "seed_scope": (
                "预测checkpoint来源seed" if report_id == "SYS5S-001" else "训练seed"
            ),
            "epochs": table_row.get("epochs", ""),
            "batch": table_row.get("batch", ""),
            "imgsz": table_row.get("imgsz", ""),
            "model": table_row.get("model", ""),
            "params": table_row.get("params", ""),
            "inference_ms": f6(table_row.get("latency_ms")),
            "precision_operating_point": precision,
            "recall_operating_point": recall,
            "numeric_source": "",
            "status": table_row.get("status", ""),
            "comparison_boundary": (
                "仅模块级联原型证据；MSE/MAE 不与检测 mAP 横向比较。"
                if report_id == "SYS5S-001"
                else "检测实验同数据、同45类、同测试划分。"
            ),
            "notes": item["notes"],
        }

        if report_id in stage5_metrics:
            metric = stage5_metrics[report_id]
            base.update(
                {
                    "best_epoch": metric["best_epoch"],
                    "val_map50": f6(metric["val_map50"]),
                    "val_map50_95": f6(metric["val_map50_95"]),
                    "test_map50": f6(metric["test_map50"]),
                    "test_map50_95": f6(metric["test_map50_95"]),
                    "test_map_small": f6(metric["test_map_small"]),
                    "test_map_medium": f6(metric["test_map_medium"]),
                    "test_map_large": f6(metric["test_map_large"]),
                    "test_mar100": f6(metric["test_mar100"]),
                    "mse": "",
                    "mae": "",
                    "numeric_source": rel_path(STAGE5_METRICS),
                }
            )
        else:
            verification = forecast["full_test_checkpoint_verification"]
            base.update(
                {
                    "best_epoch": "",
                    "val_map50": "",
                    "val_map50_95": "",
                    "test_map50": "",
                    "test_map50_95": "",
                    "test_map_small": "",
                    "test_map_medium": "",
                    "test_map_large": "",
                    "test_mar100": "",
                    "mse": f"{verification['mse']:.9f}",
                    "mae": f"{verification['mae']:.9f}",
                    "numeric_source": rel_path(FORECAST_RESULT),
                }
            )
        rows.append(base)

    fields = [
        "registry_experiment_id",
        "experiment_id",
        "role",
        "comparable_group",
        "stage",
        "seed",
        "seed_scope",
        "epochs",
        "batch",
        "imgsz",
        "model",
        "params",
        "best_epoch",
        "val_map50",
        "val_map50_95",
        "test_map50",
        "test_map50_95",
        "test_map_small",
        "test_map_medium",
        "test_map_large",
        "test_mar100",
        "precision_operating_point",
        "recall_operating_point",
        "inference_ms",
        "mse",
        "mae",
        "numeric_source",
        "status",
        "comparison_boundary",
        "notes",
    ]
    write_csv(FINAL_TABLE_CSV, rows, fields)
    write_final_table_markdown(rows)
    return rows


def write_final_table_markdown(rows: list[dict[str, str]]) -> None:
    lines = [
        "# 阶段6最终主表",
        "",
        f"- 冻结版本：`{FREEZE_VERSION}`",
        f"- 最终主模型：`{MAIN_MODEL}`",
        "- 状态：已冻结；阶段7报告和PPT只能引用本表或本表指向的原始证据。",
        "- 显示规则：检测指标保留6位小数；未计算或跨域不适用的字段写 `—`。",
        "",
        "## 一、检测与模块级联结果",
        "",
        "| 实验ID | 角色 | Seed | Epoch / 最佳验证轮 | Val mAP50-95 | Test mAP50 | Test mAP50-95 | Test small | Test medium | Test large | Test AR100 | 参数量 | 推理 ms/图 | Flow MSE | Flow MAE |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        epoch = (
            f"{row['epochs']} / {row['best_epoch']}"
            if row["best_epoch"]
            else "0 / —"
        )
        values = [
            row["experiment_id"],
            row["role"],
            row["seed"],
            epoch,
            row["val_map50_95"] or "—",
            row["test_map50"] or "—",
            row["test_map50_95"] or "—",
            row["test_map_small"] or "—",
            row["test_map_medium"] or "—",
            row["test_map_large"] or "—",
            row["test_mar100"] or "—",
            row["params"] or "—",
            row["inference_ms"] or "—",
            row["mse"] or "—",
            row["mae"] or "—",
        ]
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## 二、统一口径",
            "",
            "1. `Val` 是训练过程中用于选择 `best.pt` 的最佳验证轮指标，来源是同轮 `results.csv` 和训练汇总，不是测试集。",
            "2. `Test mAP` 默认指对保存的测试原始预测用 `pycocotools` 独立复算的结果；YOLOv10 控制台值只作一致性校验，不是主数字。",
            "3. 检测训练只做一次正式测试前向；本表不重新推理、不重新训练、不修改权重。",
            "4. 训练损失、验证损失和测试损失分表登记。现有测试评估没有产出数值型测试损失，因此主表不填测试损失。",
            "5. `SYS5S-001` 的 MSE/MAE 是复用流量 checkpoint 的完整测试复算指标，只证明可加载与可复算，不能与检测 mAP 比较，也不构成端到端性能。",
            "6. 百分比必须同时写明分子、分母和比较对象；登记见 `百分比口径登记.csv`。",
            "",
            "## 三、报告引用约束",
            "",
            "- 最终模型：`BASELINE-001`。",
            "- P2 保留为负结果/权衡实验，不写成最终模型或稳定总体提升。",
            "- 两 seed 均值与标准差只在报告中保留两位标准差说明；原始精确值仍以证据 CSV/JSON 为准。",
            "- 报告与PPT中如出现本表之外的数字，必须回到原始证据补登记并提升冻结版本。",
            "",
            "## 四、机器可读文件",
            "",
            "- `最终主表.csv`：统一数字主表。",
            "- `实验证据索引.csv`：每个实验 ID 的配置、日志、指标、checkpoint、预测路径。",
            "- `损失口径登记.csv`：训练/验证损失与测试损失缺失说明。",
            "- `百分比口径登记.csv`：百分数分子、分母、比较对象。",
            "- `证据SHA256清单.csv`：数据、权重、脚本、日志摘要、预测和证据哈希。",
        ]
    )
    with FINAL_TABLE_MD.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def build_loss_registry() -> None:
    metric_rows = build_metric_map()
    rows: list[dict[str, Any]] = []
    for item in EXPERIMENTS:
        report_id = item["report_id"]
        if report_id == "SYS5S-001":
            forecast = load_json(FORECAST_RESULT)
            verification = forecast["full_test_checkpoint_verification"]
            rows.append(
                {
                    "metric_domain": "flow_forecasting",
                    "experiment_id": report_id,
                    "results_csv": "",
                    "best_epoch": "",
                    **{column: "" for column in LOSS_COLUMNS},
                    "all_epochs_finite": "",
                    "train_loss_status": "zero_retraining_not_applicable",
                    "validation_loss_status": "not_generated_by_stage5s",
                    "test_loss_status": (
                        "full_test_checkpoint_verification_mse_mae"
                    ),
                    "test_mse": f"{verification['mse']:.12f}",
                    "test_mae": f"{verification['mae']:.12f}",
                    "test_log_path": item["test_log"],
                    "test_log_marker": "full test: mse=",
                    "source_path": rel_path(FORECAST_RESULT),
                    "notes": (
                        "MSE/MAE是流量预测测试复算指标；不是检测损失，"
                        "不得与训练/验证损失或检测mAP混用。"
                    ),
                }
            )
            continue

        results_path = as_path(item["results_csv"])
        raw_rows = load_csv(results_path)
        normalized_rows = [
            {str(key).strip(): str(value).strip() for key, value in row.items()}
            for row in raw_rows
        ]
        best_epoch = int(float(metric_rows[report_id]["best_epoch"]))
        best_row = next(
            row
            for row in normalized_rows
            if int(float(row["epoch"])) == best_epoch
        )
        all_finite = all(
            math.isfinite(float(row[column]))
            for row in normalized_rows
            for column in LOSS_COLUMNS
        )
        rows.append(
            {
                "metric_domain": "object_detection",
                "experiment_id": report_id,
                "results_csv": rel_path(results_path),
                "best_epoch": best_epoch,
                **{
                    column: best_row[column]
                    for column in LOSS_COLUMNS
                },
                "all_epochs_finite": all_finite,
                "train_loss_status": "recorded_per_epoch_in_results_csv",
                "validation_loss_status": "recorded_per_epoch_in_results_csv",
                "test_loss_status": "not_generated_by_test_eval",
                "test_mse": "",
                "test_mae": "",
                "test_log_path": rel_path(as_path(item["test_log"])),
                "test_log_marker": "0.0ms loss",
                "source_path": rel_path(results_path),
                "notes": (
                    "损失列出最佳验证轮的分量；om/oo分别对应不同匹配分支。"
                    "测试控制台仅记录耗时且为0.0ms loss，未产出测试损失。"
                ),
            }
        )

    fields = [
        "metric_domain",
        "experiment_id",
        "results_csv",
        "best_epoch",
        *LOSS_COLUMNS,
        "all_epochs_finite",
        "train_loss_status",
        "validation_loss_status",
        "test_loss_status",
        "test_mse",
        "test_mae",
        "test_log_path",
        "test_log_marker",
        "source_path",
        "notes",
    ]
    write_csv(LOSS_REGISTRY, rows, fields)


def build_percentage_registry() -> None:
    analysis = load_json(ANALYSIS_SUMMARY)
    comparison = load_json(STAGE5_COMPARISON)
    forecast = load_json(FORECAST_RESULT)
    table_rows = build_table_map()
    rows: list[dict[str, Any]] = []

    def add_ratio(
        metric_id: str,
        domain: str,
        split: str,
        experiment_or_group: str,
        numerator: float,
        denominator: float,
        unit: str,
        comparison_object: str,
        statement: str,
        source: str,
    ) -> None:
        value = numerator / denominator
        rows.append(
            {
                "metric_id": metric_id,
                "domain": domain,
                "split": split,
                "experiment_or_group": experiment_or_group,
                "numerator": f"{numerator:.12g}",
                "denominator": f"{denominator:.12g}",
                "value_fraction": f"{value:.12g}",
                "report_display": (
                    f"{value * 100:.4f}%"
                    if unit == "percent"
                    else f"{value:.6f} {unit}"
                ),
                "unit": unit,
                "comparison_object": comparison_object,
                "statement": statement,
                "source_path": source,
            }
        )

    test_snapshot = analysis["mechanism_snapshot"]["test"]
    val_snapshot = analysis["mechanism_snapshot"]["val"]
    add_ratio(
        "DIAG_TEST_SMALL_GT_SHARE",
        "object_detection",
        "test",
        "BASELINE-001",
        3302,
        7706,
        "percent",
        "全部7,706个测试GT框",
        "测试集小目标的真实框占比为42.8497%。",
        rel_path(ANALYSIS_SUMMARY),
    )
    add_ratio(
        "DIAG_TEST_SMALL_MISS_SHARE",
        "object_detection",
        "test",
        "BASELINE-001",
        1634,
        2325,
        "percent",
        "全部2,325个测试漏检",
        "测试集漏检中有70.2796%来自小目标。",
        rel_path(ANALYSIS_SUMMARY),
    )
    add_ratio(
        "DIAG_VAL_SMALL_GT_SHARE",
        "object_detection",
        "val",
        "BASELINE-001",
        720,
        1553,
        "percent",
        "全部1,553个验证GT框",
        "验证集小目标的真实框占比为46.3619%。",
        rel_path(ANALYSIS_SUMMARY),
    )
    add_ratio(
        "DIAG_VAL_SMALL_MISS_SHARE",
        "object_detection",
        "val",
        "BASELINE-001",
        361,
        479,
        "percent",
        "全部479个验证漏检",
        "验证集漏检中有75.3653%来自小目标。",
        rel_path(ANALYSIS_SUMMARY),
    )

    paired = {int(item["seed"]): item for item in comparison["paired_deltas"]}
    for seed in (2026, 2027):
        delta = float(paired[seed]["delta_test_map_small"])
        rows.append(
            {
                "metric_id": f"P2_SMALL_AP_GAIN_PP_S{seed}",
                "domain": "object_detection",
                "split": "test",
                "experiment_or_group": (
                    "P2-001 vs BASELINE-001"
                    if seed == 2026
                    else "P2-002-S2027 vs BASELINE-002-S2027"
                ),
                "numerator": f"{delta:.12g}",
                "denominator": "1",
                "value_fraction": f"{delta:.12g}",
                "report_display": f"{delta:.6f} 十进制度量点",
                "unit": "decimal_metric_point",
                "comparison_object": "同seed P2减去对应基线",
                "statement": (
                    f"seed {seed} 的测试small mAP50-95提高"
                    f"{delta:.6f}（即{delta * 100:.4f}个百分点）。"
                ),
                "source_path": rel_path(STAGE5_COMPARISON),
            }
        )

    mean_delta = float(
        comparison["interpretation"]["p2_small_mean_delta_map50_95"]
    )
    delta_sum = sum(
        float(comparison["paired_deltas"][index]["delta_test_map_small"])
        for index in range(2)
    )
    rows.append(
        {
            "metric_id": "P2_SMALL_AP_MEAN_GAIN_PP_N2",
            "domain": "object_detection",
            "split": "test",
            "experiment_or_group": "P2 two-seed mean",
            "numerator": f"{delta_sum:.12g}",
            "denominator": "2",
            "value_fraction": f"{mean_delta:.12g}",
            "report_display": f"{mean_delta:.6f} 十进制度量点",
            "unit": "decimal_metric_point",
            "comparison_object": "两个配对seed的P2减基线差值均值",
            "statement": (
                f"P2相对配对基线的small mAP50-95平均提高"
                f"{mean_delta:.6f}（即{mean_delta * 100:.4f}个百分点，n=2）。"
            ),
            "source_path": rel_path(STAGE5_COMPARISON),
        }
    )

    baseline_latency = float(table_rows["BASE_YOLOV10N_TT100K45_S2026"]["latency_ms"])
    p2_latency = float(table_rows["P2-001"]["latency_ms"])
    add_ratio(
        "P2_LATENCY_RATIO_S2026",
        "object_detection",
        "test",
        "P2-001 / BASELINE-001",
        p2_latency,
        baseline_latency,
        "ratio",
        "BASELINE-001单图推理时延4.2 ms",
        "P2-001单图推理时延约为基线的2.900142倍。",
        rel_path(EXPERIMENT_TABLE),
    )

    summary = forecast["cascade_example"]["summary"]
    with np.load(CASCADE_PREDICTIONS, allow_pickle=False) as data:
        baseline_scale = float(np.mean(np.abs(data["baseline_prediction"])) + 1e-12)
    mean_abs_delta = float(summary["mean_abs_prediction_delta"])
    add_ratio(
        "SYS5S_REL_MEAN_ABS_PREDICTION_DELTA",
        "flow_forecasting",
        "single_window_diagnostic",
        "SYS5S-001",
        mean_abs_delta,
        baseline_scale,
        "percent",
        "注入前预测绝对值的均值",
        "单窗口注入造成相对平均绝对预测差1.4722%，只作接口传播证据。",
        rel_path(FORECAST_RESULT),
    )

    fields = [
        "metric_id",
        "domain",
        "split",
        "experiment_or_group",
        "numerator",
        "denominator",
        "value_fraction",
        "report_display",
        "unit",
        "comparison_object",
        "statement",
        "source_path",
    ]
    write_csv(PERCENT_REGISTRY, rows, fields)

    expected = {
        "DIAG_TEST_SMALL_GT_SHARE": test_snapshot["small_ground_truth_share"],
        "DIAG_TEST_SMALL_MISS_SHARE": test_snapshot["small_share_of_all_misses"],
        "DIAG_VAL_SMALL_GT_SHARE": val_snapshot["small_ground_truth_share"],
        "DIAG_VAL_SMALL_MISS_SHARE": val_snapshot["small_share_of_all_misses"],
    }
    actual = {row["metric_id"]: float(row["value_fraction"]) for row in rows}
    for metric_id, value in expected.items():
        if abs(actual[metric_id] - float(value)) > 1e-12:
            raise RuntimeError(f"百分比分母复算不一致: {metric_id}")


def build_change_register() -> None:
    rows = [
        {
            "freeze_version": FREEZE_VERSION,
            "change_id": "FREEZE-000",
            "change_type": "initial_freeze",
            "numbers_changed": "否",
            "reason": "首次建立阶段6统一数字来源，未修改既有实验结果。",
            "evidence": rel_path(FINAL_TABLE_CSV),
            "review_status": "approved_by_stage6_audit",
        }
    ]
    fields = list(rows[0])
    write_csv(CHANGE_REGISTER, rows, fields)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def generate_figures(
    output_dir: Path,
    metric_map: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    configure_plotting()
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_map = metric_map or build_metric_map()
    seed_pairs = ["BASELINE-001", "BASELINE-002-S2027", "P2-001", "P2-002-S2027"]
    labels = ["基线两seed", "P2两seed", "48轮预算对照"]
    colors = ["#3A6EA5", "#C44E52", "#7A7A7A"]
    metric_ids = ["test_map50_95", "test_map_small"]
    metric_labels = ["测试mAP50-95", "测试small mAP50-95"]

    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for experiment_id in seed_pairs:
        for metric_id in metric_ids:
            groups[(experiment_id, metric_id)].append(
                float(metric_map[experiment_id][metric_id])
            )

    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(metric_ids))
    width = 0.25
    group_specs = [
        ("baseline_seed", labels[0], ["BASELINE-001", "BASELINE-002-S2027"]),
        ("p2_seed", labels[1], ["P2-001", "P2-002-S2027"]),
        (
            "budget",
            labels[2],
            ["BASELINE-BUDGET48-S2026"],
        ),
    ]
    for offset_index, (_, label, ids) in enumerate(group_specs):
        means = []
        errors = []
        for metric_id in metric_ids:
            values = [
                float(metric_map[experiment_id][metric_id]) for experiment_id in ids
            ]
            means.append(mean(values))
            errors.append(stdev(values) if len(values) > 1 else 0.0)
        positions = x + (offset_index - 1) * width
        ax.bar(
            positions,
            means,
            width=width,
            label=label,
            color=colors[offset_index],
            alpha=0.88,
            yerr=errors,
            capsize=4,
        )
        if len(ids) > 1:
            for position, metric_id in zip(positions, metric_ids):
                values = [
                    float(metric_map[experiment_id][metric_id]) for experiment_id in ids
                ]
                ax.scatter(
                    [position] * len(values),
                    values,
                    s=26,
                    color="black",
                    zorder=4,
                )
    ax.set_xticks(x, metric_labels)
    ax.set_ylim(0, 0.45)
    ax.set_ylabel("指标值（0–1）")
    ax.set_title("检测主结果：总体与小目标测试AP")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig1 = output_dir / "fig1_detection_main_results.png"
    fig.savefig(
        fig1,
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": FREEZE_VERSION},
    )
    plt.close(fig)

    baseline_results = as_path(EXPERIMENTS[0]["results_csv"])
    raw_rows = load_csv(baseline_results)
    rows = [
        {str(key).strip(): str(value).strip() for key, value in row.items()}
        for row in raw_rows
    ]
    epochs = np.array([int(float(row["epoch"])) for row in rows])
    components = [
        ("box", "边界框损失", ["train/box_om", "val/box_om", "train/box_oo", "val/box_oo"]),
        ("cls", "分类损失", ["train/cls_om", "val/cls_om", "train/cls_oo", "val/cls_oo"]),
        ("dfl", "DFL损失", ["train/dfl_om", "val/dfl_om", "train/dfl_oo", "val/dfl_oo"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
    line_styles = ["-", "--", ":", "-."]
    line_labels = ["训练-om", "验证-om", "训练-oo", "验证-oo"]
    for ax, (_, title, columns) in zip(axes, components):
        for column, linestyle, line_label in zip(columns, line_styles, line_labels):
            ax.plot(
                epochs,
                [float(row[column]) for row in rows],
                linestyle=linestyle,
                linewidth=1.6,
                label=line_label,
            )
        ax.axvline(39, color="#777777", linewidth=1, alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("损失值")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("BASELINE-001 训练与验证损失（来自results.csv，不含测试损失）")
    fig.tight_layout()
    fig2 = output_dir / "fig2_baseline_training_validation_losses.png"
    fig.savefig(
        fig2,
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": FREEZE_VERSION},
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.4))
    size_metrics = ["test_map_small", "test_map_medium", "test_map_large"]
    size_labels = ["small", "medium", "large"]
    positions = np.arange(len(size_metrics))
    for index, (ids, label, color) in enumerate(
        [
            (
                ["BASELINE-001", "BASELINE-002-S2027"],
                "基线两seed",
                "#3A6EA5",
            ),
            (
                ["P2-001", "P2-002-S2027"],
                "P2两seed",
                "#C44E52",
            ),
        ]
    ):
        means = [
            mean(float(metric_map[experiment_id][metric_id]) for experiment_id in ids)
            for metric_id in size_metrics
        ]
        errors = [
            stdev(float(metric_map[experiment_id][metric_id]) for experiment_id in ids)
            for metric_id in size_metrics
        ]
        bar_positions = positions + (index - 0.5) * 0.34
        ax.bar(
            bar_positions,
            means,
            width=0.34,
            yerr=errors,
            capsize=4,
            color=color,
            label=label,
            alpha=0.88,
        )
        for bar_position, metric_id in zip(bar_positions, size_metrics):
            values = [float(metric_map[experiment_id][metric_id]) for experiment_id in ids]
            ax.scatter(
                [bar_position] * len(values),
                values,
                color="black",
                s=24,
                zorder=4,
            )
    ax.set_xticks(positions, size_labels)
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("测试mAP50-95（0–1）")
    ax.set_title("测试集COCO尺寸分桶AP：两seed均值±标准差")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig3 = output_dir / "fig3_test_size_bucket_ap.png"
    fig.savefig(
        fig3,
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": FREEZE_VERSION},
    )
    plt.close(fig)

    with np.load(CASCADE_PREDICTIONS, allow_pickle=False) as data:
        difference = np.abs(
            data["injected_prediction"] - data["baseline_prediction"]
        )
    fig, ax = plt.subplots(figsize=(10, 5.4))
    image = ax.imshow(difference, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xlabel("PEMS08传感器通道")
    ax.set_ylabel("预测时间步")
    ax.set_title("SYS5S-001单窗口预测绝对差（接口诊断，不表示精度提升）")
    fig.colorbar(image, ax=ax, label="|注入后预测 - 注入前预测|")
    fig.tight_layout()
    fig4 = output_dir / "fig4_sys5s_cascade_difference.png"
    fig.savefig(
        fig4,
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": FREEZE_VERSION},
    )
    plt.close(fig)

    specs = [
        {
            "figure_id": "FIG-1",
            "filename": fig1.name,
            "title": "检测主结果：总体与小目标测试AP",
            "report_use": "主结果图；基线两seed、P2两seed与48轮预算对照同图，不做跨域比较。",
            "data_sources": (
                f"{rel_path(STAGE5_METRICS)};{rel_path(STAGE5_COMPARISON)}"
            ),
        },
        {
            "figure_id": "FIG-2",
            "filename": fig2.name,
            "title": "BASELINE-001训练与验证损失",
            "report_use": "训练过程图；明确标注不含测试损失。",
            "data_sources": rel_path(baseline_results),
        },
        {
            "figure_id": "FIG-3",
            "filename": fig3.name,
            "title": "测试集COCO尺寸分桶AP",
            "report_use": "小目标机制与总体权衡图。",
            "data_sources": rel_path(STAGE5_METRICS),
        },
        {
            "figure_id": "FIG-4",
            "filename": fig4.name,
            "title": "SYS5S-001单窗口预测绝对差",
            "report_use": "模块级联接证据；禁止写成端到端精度或性能提升。",
            "data_sources": rel_path(CASCADE_PREDICTIONS),
        },
    ]
    manifest_rows: list[dict[str, str]] = []
    for spec in specs:
        path = output_dir / spec["filename"]
        with Image.open(path) as image:
            width, height = image.size
        manifest_rows.append(
            {
                **spec,
                "path": rel_path(path),
                "sha256": sha256_file(path, show_progress=False),
                "width_px": str(width),
                "height_px": str(height),
                "regeneration_command": (
                    "02_代码\\环境\\venv\\Scripts\\python.exe "
                    "02_代码\\阶段6\\build_stage6_freeze.py --figures-only"
                ),
            }
        )
    return manifest_rows


def build_figure_manifest() -> list[dict[str, str]]:
    rows = generate_figures(FIGURE_ROOT)
    fields = [
        "figure_id",
        "filename",
        "title",
        "report_use",
        "data_sources",
        "path",
        "sha256",
        "width_px",
        "height_px",
        "regeneration_command",
    ]
    write_csv(FIGURE_MANIFEST, rows, fields)
    return rows


def build_metadata(figures: list[dict[str, str]]) -> None:
    payload = {
        "freeze_version": FREEZE_VERSION,
        "generated_at": now_iso(),
        "stage": "6",
        "stage_name": "证据冻结",
        "stage5s_baseline_commit": STAGE5S_COMMIT,
        "stage5s_tag": STAGE5S_TAG,
        "main_model": MAIN_MODEL,
        "registry_ids": [item["registry_id"] for item in EXPERIMENTS],
        "canonical_experiment_ids": [item["report_id"] for item in EXPERIMENTS],
        "canonical_detection_metrics_source": rel_path(STAGE5_METRICS),
        "canonical_system_metrics_source": rel_path(FORECAST_RESULT),
        "test_policy": {
            "detector_test_inference_per_model": 1,
            "stage6_retraining": False,
            "stage6_test_inference": False,
            "stage6_weight_or_checkpoint_modification": False,
            "stage5s_reevaluation": False,
        },
        "loss_policy": {
            "training_loss_source": "results.csv:train/*",
            "validation_loss_source": "results.csv:val/*",
            "test_loss_status": "未由现有检测测试流程生成或持久化",
            "zero_ms_loss_meaning": "控制台中的 loss 计时为0.0ms，不是损失数值。",
        },
        "excluded_inputs": [
            {
                "path": "03_实验/数据/原始/tt100k_2021.zip.partial-unverified",
                "reason": "未验证的旧 partial 下载，不参与阶段6冻结。",
            }
        ],
        "figures": figures,
        "comparison_boundary": (
            "检测mAP与SYS5S流量MSE/MAE分域保存；"
            "SYS5S只证明离线模块接口与数据流，不是端到端性能。"
        ),
    }
    write_json(FREEZE_METADATA, payload)


def run_git(*args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip()


def collect_assets() -> list[dict[str, str]]:
    assets: list[dict[str, str]] = []
    seen_paths: set[str] = set()

    def add(
        category: str,
        asset_id: str,
        path: str | Path,
        asset_type: str = "file",
        notes: str = "",
        repository_policy: str = "",
    ) -> None:
        resolved = as_path(path)
        key = str(resolved.resolve()).lower()
        if key in seen_paths:
            return
        seen_paths.add(key)
        assets.append(
            {
                "category": category,
                "asset_id": asset_id,
                "path": rel_path(resolved),
                "asset_type": asset_type,
                "notes": notes,
                "repository_policy": repository_policy,
            }
        )

    data_manifest_root = (
        PROJECT_ROOT / "03_实验" / "datasets" / "TT100K" / "yolo"
    )
    add(
        "data",
        "TT100K_RAW_ARCHIVE",
        TT100K_RAW,
        notes="19,152,969,603字节；仅本地保留，远端使用SHA-256定位。",
        repository_policy="local_only_hash_recorded",
    )
    add(
        "data",
        "PEMS08",
        PEMS08,
        notes="第一次大作业复用数据；外部路径资产。",
        repository_policy="external_local_only",
    )
    for name in [
        "class_selection.json",
        "dataset_manifest.json",
        "dataset_statistics.csv",
        "tt100k.yaml",
        "manifests/class_to_id.json",
        "manifests/dataset_summary.json",
        "manifests/leakage_exclusions.json",
        "manifests/split_manifest.json",
        "stats/class_statistics.csv",
    ]:
        add(
            "data",
            "TT100K_MANIFEST_" + name.replace("/", "_"),
            data_manifest_root / name,
            notes="固定数据版本、划分或类别统计。",
            repository_policy="commit",
        )

    add(
        "weights",
        "YOLOV10N_INITIAL_PT",
        PROJECT_ROOT / "02_代码" / "权重" / "yolov10n.pt",
        notes="训练初始化载体。",
        repository_policy="local_only_hash_recorded",
    )
    add(
        "weights",
        "YOLOV10N_HF_SAFETENSORS",
        PROJECT_ROOT / "02_代码" / "权重" / "yolov10n_hf" / "model.safetensors",
        notes="官方来源权重载体。",
        repository_policy="local_only_hash_recorded",
    )
    add(
        "weights",
        "YOLOV10N_HF_CONFIG",
        PROJECT_ROOT / "02_代码" / "权重" / "yolov10n_hf" / "config.json",
        notes="官方权重配置。",
        repository_policy="commit",
    )
    for item in EXPERIMENTS[:-1]:
        add(
            "weights",
            f"{item['report_id']}_BEST_PT",
            as_path(item["checkpoint"]),
            notes="对应实验的最佳验证checkpoint；阶段6不修改。",
            repository_policy="local_only_hash_recorded",
        )
    add(
        "weights",
        "SYS5S_FLOW_CHECKPOINT",
        FLOW_CHECKPOINT,
        notes="第一次大作业TA-iTransformer checkpoint；零重训复用。",
        repository_policy="external_local_only",
    )

    for phase in ["阶段2", "阶段3", "阶段4", "阶段5", "阶段5S", "阶段6"]:
        for path in sorted((PROJECT_ROOT / "02_代码" / phase).rglob("*.py")):
            add(
                "script",
                "PY_" + rel_path(path).replace("/", "_"),
                path,
                notes="关键执行、分析或审计脚本。",
                repository_policy="commit",
            )
        for path in sorted((PROJECT_ROOT / "02_代码" / phase).rglob("*.ps1")):
            add(
                "script",
                "PS1_" + rel_path(path).replace("/", "_"),
                path,
                notes="关键PowerShell入口。",
                repository_policy="commit",
            )
    for path in sorted((PROJECT_ROOT / "02_代码" / "阶段4" / "configs").glob("*.yaml")):
        add(
            "script",
            "ARCH_" + path.name,
            path,
            notes="P2/基线架构定义。",
            repository_policy="commit",
        )

    for path in sorted((PROJECT_ROOT / "03_实验" / "configs").glob("*.yaml")):
        add(
            "config",
            "EXP_" + path.name,
            path,
            notes="实验锁定配置。",
            repository_policy="commit",
        )

    log_roots = [
        PROJECT_ROOT / "03_实验" / "logs" / "BASELINE-001",
        PROJECT_ROOT / "03_实验" / "logs" / "P2-001",
        PROJECT_ROOT / "03_实验" / "logs" / "STAGE5",
        PROJECT_ROOT / "03_实验" / "logs" / "STAGE5S",
    ]
    for root in log_roots:
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            add(
                "log_summary",
                "LOG_" + rel_path(path).replace("/", "_"),
                path,
                notes="本地保留的原始/摘要日志。",
                repository_policy="local_only_hash_recorded",
            )

    for item in EXPERIMENTS[:-1]:
        for key in ["predictions", "prediction_support"]:
            path = as_path(item[key])
            add(
                "prediction",
                f"{item['report_id']}_{key}",
                path,
                notes="COCO格式测试预测；用于主指标独立复算。",
                repository_policy="local_only_hash_recorded",
            )
        add(
            "prediction",
            f"{item['report_id']}_LABELS_DIRECTORY",
            as_path(item["prediction_labels"]),
            asset_type="directory",
            notes="逐图测试预测标签目录的路径+内容聚合摘要。",
            repository_policy="local_only_hash_recorded",
        )
    add(
        "prediction",
        "SYS5S_CASCADE_PREDICTIONS",
        CASCADE_PREDICTIONS,
        notes="单窗口基准/注入预测数组；只冻结，不重算。",
        repository_policy="commit",
    )

    evidence_roots = [
        PROJECT_ROOT / "01_立项" / f"阶段{stage}_证据"
        for stage in ["2", "3", "4", "5", "5S"]
    ]
    for root in evidence_roots:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            add(
                "evidence",
                "EVID_" + rel_path(path).replace("/", "_"),
                path,
                notes="既有阶段证据；阶段6只冻结，不重评。",
                repository_policy="commit",
            )
    for path in [
        PROJECT_ROOT / "01_立项" / "阶段总结" / f"阶段{stage}_总结.md"
        for stage in ["2", "3", "4", "5", "5S"]
    ]:
        add(
            "evidence",
            "SUMMARY_" + path.name,
            path,
            notes="阶段结论与证据引用。",
            repository_policy="commit",
        )

    diagnosis_root = (
        PROJECT_ROOT / "03_实验" / "诊断" / "BASELINE-001" / "analysis"
    )
    for relative in [
        "analysis_summary.json",
        "size_bucket_metrics.csv",
        "test_pred_events.csv",
        "figures/case_montage.png",
        "figures/class_ap_vs_support.png",
        "figures/size_bucket_recall.png",
        "figures/threshold_sweep.png",
        "figures/top_confusion_pairs.png",
    ]:
        path = diagnosis_root / relative
        add(
            "evidence",
            "DIAG_" + relative.replace("/", "_"),
            path,
            notes="阶段3诊断与SYS5S检测事件来源。",
            repository_policy="commit",
        )

    for path in [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "01_立项" / "项目信息卡.md",
        PROJECT_ROOT / "01_立项" / "阶段门槛与验收清单.md",
        EXPERIMENT_TABLE,
        STAGE5_METRICS,
        STAGE5_COMPARISON,
    ]:
        add(
            "metadata",
            "META_" + path.name,
            path,
            notes="阶段6事实源或项目状态文件。",
            repository_policy="commit",
        )

    for path in [
        FINAL_TABLE_CSV,
        FINAL_TABLE_MD,
        EXPERIMENT_INDEX,
        LOSS_REGISTRY,
        PERCENT_REGISTRY,
        FIGURE_MANIFEST,
        CHANGE_REGISTER,
        FREEZE_METADATA,
        VERSION_RULES,
        EVIDENCE_ROOT / "README.md",
    ]:
        add(
            "stage6_output",
            "STAGE6_" + path.name,
            path,
            notes="阶段6冻结层输出。",
            repository_policy="commit",
        )
    for path in sorted(FIGURE_ROOT.glob("*.png")):
        add(
            "stage6_output",
            "STAGE6_FIGURE_" + path.name,
            path,
            notes="可由阶段6脚本从冻结数据离线重绘图。",
            repository_policy="commit",
        )
    return assets


def build_hash_manifest() -> list[dict[str, str]]:
    assets = collect_assets()
    rows: list[dict[str, str]] = []
    for index, asset in enumerate(assets, start=1):
        path = as_path(asset["path"])
        exists = path.exists()
        digest = ""
        size = ""
        file_count = ""
        status = "missing"
        if exists and asset["asset_type"] == "directory" and path.is_dir():
            digest, file_count_value, size_value = sha256_directory(path)
            file_count = str(file_count_value)
            size = str(size_value)
            status = "present"
        elif exists and path.is_file():
            size = str(path.stat().st_size)
            digest = sha256_file(path)
            file_count = "1"
            status = "present"
        rows.append(
            {
                "sequence": str(index),
                "category": asset["category"],
                "asset_id": asset["asset_id"],
                "path": asset["path"],
                "asset_type": asset["asset_type"],
                "status": status,
                "size_bytes": size,
                "file_count": file_count,
                "sha256": digest,
                "hash_algorithm": "SHA-256",
                "repository_policy": asset["repository_policy"],
                "archive_status": (
                    "hash_recorded; binary_not_embedded"
                    if asset["repository_policy"] != "commit"
                    else "hash_recorded; content_or_pointer_committed"
                ),
                "notes": asset["notes"],
            }
        )
        if index % 25 == 0 or index == len(assets):
            print(f"已计算证据哈希 {index}/{len(assets)}")
    fields = [
        "sequence",
        "category",
        "asset_id",
        "path",
        "asset_type",
        "status",
        "size_bytes",
        "file_count",
        "sha256",
        "hash_algorithm",
        "repository_policy",
        "archive_status",
        "notes",
    ]
    write_csv(HASH_MANIFEST, rows, fields)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="生成阶段6证据冻结层")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--tables", action="store_true", help="生成登记表和主图")
    mode.add_argument("--figures-only", action="store_true", help="仅重绘主图")
    mode.add_argument("--manifest", action="store_true", help="生成SHA-256清单")
    mode.add_argument("--all", action="store_true", help="生成全部冻结层")
    args = parser.parse_args()
    mode_name = (
        "tables"
        if args.tables
        else "figures"
        if args.figures_only
        else "manifest"
        if args.manifest
        else "all"
    )
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)

    if mode_name in {"tables", "all"}:
        build_experiment_index()
        build_final_table()
        build_loss_registry()
        build_percentage_registry()
        build_change_register()
        figures = build_figure_manifest()
        build_metadata(figures)
        print(f"登记表与主图已生成: {EVIDENCE_ROOT}")
    elif mode_name == "figures":
        figures = build_figure_manifest()
        build_metadata(figures)
        print(f"主图已离线重绘: {FIGURE_ROOT}")

    if mode_name in {"manifest", "all"}:
        rows = build_hash_manifest()
        print(f"SHA-256清单已生成: {HASH_MANIFEST} ({len(rows)}项)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
