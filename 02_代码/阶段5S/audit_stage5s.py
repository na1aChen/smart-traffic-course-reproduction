from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = PROJECT_ROOT / "01_立项" / "阶段5S_证据"
SUMMARY_PATH = PROJECT_ROOT / "01_立项" / "阶段总结" / "阶段5S_总结.md"
AUDIT_PATH = EVIDENCE_DIR / "阶段5S审计.json"
EXPERIMENT_TABLE = PROJECT_ROOT / "03_实验" / "实验总表.csv"
RUNTIME_LOG = PROJECT_ROOT / "03_实验" / "logs" / "STAGE5S" / "SYS5S-001_run.log"


def utc_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON 顶层不是对象: {path}")
    return value


def audit() -> tuple[dict[str, bool], dict[str, Any]]:
    required_files = {
        "preregistration": EVIDENCE_DIR / "阶段5S预注册与接口设计.md",
        "result_analysis": EVIDENCE_DIR / "阶段5S_结果分析.md",
        "asset_check": EVIDENCE_DIR / "阶段5S_资产检查.json",
        "forecast_result": EVIDENCE_DIR / "forecast_result.json",
        "detector_bins": EVIDENCE_DIR / "detector_bins.csv",
        "forecast_preview": EVIDENCE_DIR / "forecast_preview.csv",
        "predictions": EVIDENCE_DIR / "cascade_predictions.npz",
        "plot": EVIDENCE_DIR / "cascade_difference.png",
        "runner": PROJECT_ROOT / "02_代码" / "阶段5S" / "run_offline_cascade.py",
        "summary": SUMMARY_PATH,
    }
    file_checks = {name: path.is_file() for name, path in required_files.items()}
    missing_files = [str(path) for name, path in required_files.items() if not file_checks[name]]
    if missing_files:
        raise FileNotFoundError("阶段5S必需文件缺失: " + "; ".join(missing_files))

    asset = load_json(required_files["asset_check"])
    forecast = load_json(required_files["forecast_result"])

    with required_files["detector_bins"].open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        detector_rows = list(csv.DictReader(handle))
    detector_fields = set(detector_rows[0]) if detector_rows else set()
    required_detector_fields = {
        "bin",
        "image_id_min",
        "image_id_max",
        "image_count",
        "detection_count",
        "score_sum",
    }

    with np.load(required_files["predictions"], allow_pickle=False) as prediction_data:
        expected_arrays = {
            "baseline_prediction": (96, 170),
            "injected_prediction": (96, 170),
            "target": (96, 170),
            "detector_proxy": (96,),
            "clean_input": (96, 170),
            "injected_input": (96, 170),
        }
        actual_shapes = {
            name: tuple(prediction_data[name].shape) for name in expected_arrays
        }
        arrays_finite = all(
            bool(np.isfinite(prediction_data[name]).all()) for name in expected_arrays
        )

    with EXPERIMENT_TABLE.open("r", encoding="utf-8-sig", newline="") as handle:
        experiment_rows = list(csv.DictReader(handle))
    system_row = next(
        (row for row in experiment_rows if row.get("experiment_id") == "SYS5S-001"),
        None,
    )

    checks = {
        "required_files_exist": all(file_checks.values()),
        "detector_output_structured": bool(detector_rows)
        and required_detector_fields.issubset(detector_fields),
        "detector_bins_96": len(detector_rows) == 96,
        "checkpoint_strict_load": asset.get("checkpoint", {}).get("strict_load")
        is True
        and not asset.get("checkpoint", {}).get("missing_keys")
        and not asset.get("checkpoint", {}).get("unexpected_keys"),
        "prediction_arrays_shape_and_finite": actual_shapes == expected_arrays
        and arrays_finite,
        "chain_scope_disclosed": forecast.get("chain_type")
        == "offline_module_cascade_prototype"
        and forecast.get("valid_as_real_end_to_end_traffic_validation") is False,
        "runner_checks_9_of_9": len(forecast.get("checks", {})) == 9
        and all(forecast.get("checks", {}).values()),
        "reference_metrics_reproduced": (
            forecast.get("full_test_checkpoint_verification", {}).get(
                "absolute_mse_difference", 1.0
            )
            <= 1e-4
            and forecast.get("full_test_checkpoint_verification", {}).get(
                "absolute_mae_difference", 1.0
            )
            <= 1e-4
        ),
        "experiment_table_registered": system_row is not None
        and system_row.get("status") == "passed_prototype",
        "experiment_run_log_exists": RUNTIME_LOG.is_file()
        and RUNTIME_LOG.stat().st_size > 0,
        "summary_generated": SUMMARY_PATH.stat().st_size > 0,
    }

    hashes = {
        name: sha256_file(path)
        for name, path in required_files.items()
        if path.suffix.lower() != ".npz"
    }
    hashes["predictions"] = sha256_file(required_files["predictions"])
    details = {
        "generated_at": utc_now(),
        "stage": "5S",
        "experiment_id": "SYS5S-001",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "overall_passed": all(checks.values()),
        "detector_bins": len(detector_rows),
        "prediction_shapes": actual_shapes,
        "forecast_summary": forecast.get("cascade_example", {}).get("summary"),
        "full_test_checkpoint_verification": forecast.get(
            "full_test_checkpoint_verification"
        ),
        "file_sha256": hashes,
    }
    return checks, details


def main() -> int:
    checks, details = audit()
    AUDIT_PATH.write_text(
        json.dumps(details, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print(f"audit passed: {details['passed']}/{details['total']}")
    print(f"audit file: {AUDIT_PATH}")
    return 0 if details["overall_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
