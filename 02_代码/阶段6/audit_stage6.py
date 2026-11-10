from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import build_stage6_freeze as builder
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = PROJECT_ROOT / "01_立项" / "阶段6_证据"
SUMMARY_PATH = PROJECT_ROOT / "01_立项" / "阶段总结" / "阶段6_总结.md"
AUDIT_PATH = EVIDENCE_ROOT / "阶段6审计.json"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON 顶层不是对象: {path}")
    return payload


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def git(*args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip()


def normalized_path(path: str | Path) -> str:
    return str(as_path(path).resolve()).lower()


def crlf_variant_sha256(path: Path) -> str | None:
    suffixes = {".json", ".csv", ".md", ".txt", ".yaml", ".yml", ".py", ".ps1"}
    if path.suffix.lower() not in suffixes or not path.is_file():
        return None
    payload = path.read_bytes().replace(b"\r\n", b"\n")
    variant = payload.replace(b"\n", b"\r\n")
    return hashlib.sha256(variant).hexdigest().upper()


def walk_dicts(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        found.append(value)
        for child in value.values():
            found.extend(walk_dicts(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(walk_dicts(child))
    return found


def run_audit() -> tuple[dict[str, bool], dict[str, Any]]:
    required_paths = {
        "final_table_csv": builder.FINAL_TABLE_CSV,
        "final_table_markdown": builder.FINAL_TABLE_MD,
        "experiment_index": builder.EXPERIMENT_INDEX,
        "loss_registry": builder.LOSS_REGISTRY,
        "percentage_registry": builder.PERCENT_REGISTRY,
        "figure_manifest": builder.FIGURE_MANIFEST,
        "change_register": builder.CHANGE_REGISTER,
        "hash_manifest": builder.HASH_MANIFEST,
        "metadata": builder.FREEZE_METADATA,
        "version_rules": builder.VERSION_RULES,
        "evidence_readme": EVIDENCE_ROOT / "README.md",
        "builder": PROJECT_ROOT / "02_代码" / "阶段6" / "build_stage6_freeze.py",
    }
    missing_required = [
        str(path) for path in required_paths.values() if not path.is_file()
    ]
    if missing_required:
        raise FileNotFoundError("阶段6必需文件缺失: " + "; ".join(missing_required))

    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    stage5s_tag_commit = git("rev-parse", f"{builder.STAGE5S_TAG}^{{commit}}")
    stage5s_tag_ok = stage5s_tag_commit.startswith(builder.STAGE5S_COMMIT)
    checks["stage5s_baseline_pinned"] = stage5s_tag_ok
    details["stage5s_baseline"] = {
        "expected_commit": builder.STAGE5S_COMMIT,
        "tag": builder.STAGE5S_TAG,
        "tag_commit": stage5s_tag_commit,
    }

    stage5s_audit = load_json(
        PROJECT_ROOT / "01_立项" / "阶段5S_证据" / "阶段5S审计.json"
    )
    stage5s_audit_ok = (
        stage5s_audit.get("overall_passed") is True
        and stage5s_audit.get("passed") == 11
        and stage5s_audit.get("total") == 11
    )
    checks["stage5s_audit_passed_without_reevaluation"] = stage5s_audit_ok
    details["stage5s_audit"] = {
        "overall_passed": stage5s_audit.get("overall_passed"),
        "passed": stage5s_audit.get("passed"),
        "total": stage5s_audit.get("total"),
        "stage6_action": "读取并冻结既有审计证据，未调用阶段5S运行器。",
    }

    table_rows = load_csv(builder.EXPERIMENT_TABLE)
    table_ids = {row["experiment_id"] for row in table_rows}
    expected_registry_ids = {item["registry_id"] for item in builder.EXPERIMENTS}
    expected_canonical_ids = {item["report_id"] for item in builder.EXPERIMENTS}
    index_rows = load_csv(builder.EXPERIMENT_INDEX)
    index_registry_ids = {
        row["registry_experiment_id"] for row in index_rows
    }
    index_canonical_ids = {
        row["canonical_experiment_id"] for row in index_rows
    }
    checks["experiment_ids_complete"] = (
        table_ids == expected_registry_ids
        and index_registry_ids == expected_registry_ids
        and index_canonical_ids == expected_canonical_ids
        and len(index_rows) == len(builder.EXPERIMENTS)
    )
    details["experiment_ids"] = {
        "table": sorted(table_ids),
        "index_registry": sorted(index_registry_ids),
        "index_canonical": sorted(index_canonical_ids),
    }

    existence_fields = [
        "config_exists",
        "log_exists",
        "training_summary_exists",
        "validation_metrics_exists",
        "test_metrics_exists",
        "checkpoint_paths_exist",
        "primary_prediction_exists",
        "supporting_prediction_exists",
        "prediction_labels_exists",
        "test_log_exists",
    ]
    path_failures: list[dict[str, str]] = []
    for row in index_rows:
        for field in existence_fields:
            value = row.get(field, "")
            if field in {"training_summary_exists", "validation_metrics_exists"} and value == "":
                continue
            if field in {
                "supporting_prediction_exists",
                "prediction_labels_exists",
            } and value == "":
                continue
            if value != "True":
                path_failures.append(
                    {
                        "experiment_id": row["canonical_experiment_id"],
                        "field": field,
                        "value": value,
                    }
                )
    checks["experiment_evidence_paths_complete"] = not path_failures
    details["experiment_evidence_path_failures"] = path_failures

    metric_rows = builder.build_metric_map()
    metric_mismatches: list[dict[str, Any]] = []
    for row in table_rows:
        registry_id = row["experiment_id"]
        mapping = next(
            item for item in builder.EXPERIMENTS if item["registry_id"] == registry_id
        )
        if mapping["report_id"] not in metric_rows:
            continue
        metric = metric_rows[mapping["report_id"]]
        for table_field, source_field in [
            ("mAP50", "test_map50"),
            ("mAP50_95", "test_map50_95"),
        ]:
            table_value = f"{float(row[table_field]):.6f}"
            source_value = f"{float(metric[source_field]):.6f}"
            if table_value != source_value:
                metric_mismatches.append(
                    {
                        "experiment_id": mapping["report_id"],
                        "field": table_field,
                        "experiment_table": table_value,
                        "stage5_metrics": source_value,
                    }
                )
    checks["experiment_table_metrics_consistent"] = not metric_mismatches
    details["experiment_table_metric_mismatches"] = metric_mismatches

    final_rows = load_csv(builder.FINAL_TABLE_CSV)
    final_map = {row["experiment_id"]: row for row in final_rows}
    final_mismatches: list[dict[str, Any]] = []
    for experiment_id, metric in metric_rows.items():
        row = final_map.get(experiment_id)
        if row is None:
            final_mismatches.append(
                {"experiment_id": experiment_id, "field": "row", "value": "missing"}
            )
            continue
        for field in [
            "val_map50_95",
            "test_map50",
            "test_map50_95",
            "test_map_small",
            "test_map_medium",
            "test_map_large",
            "test_mar100",
        ]:
            expected = f"{float(metric[field]):.6f}"
            if row[field] != expected:
                final_mismatches.append(
                    {
                        "experiment_id": experiment_id,
                        "field": field,
                        "final": row[field],
                        "expected": expected,
                    }
                )
    forecast = load_json(builder.FORECAST_RESULT)
    verification = forecast["full_test_checkpoint_verification"]
    sys_row = final_map.get("SYS5S-001")
    sys_expected = {
        "mse": f"{float(verification['mse']):.9f}",
        "mae": f"{float(verification['mae']):.9f}",
    }
    for field, expected in sys_expected.items():
        if sys_row is None or sys_row[field] != expected:
            final_mismatches.append(
                {
                    "experiment_id": "SYS5S-001",
                    "field": field,
                    "final": "" if sys_row is None else sys_row[field],
                    "expected": expected,
                }
            )
    checks["final_table_metrics_consistent"] = not final_mismatches
    details["final_table_metric_mismatches"] = final_mismatches

    markdown = builder.FINAL_TABLE_MD.read_text(encoding="utf-8")
    markdown_missing: list[str] = []
    for experiment_id, row in final_map.items():
        if experiment_id not in markdown:
            markdown_missing.append(experiment_id)
            continue
        for field in ["test_map50", "test_map50_95", "test_map_small"]:
            if field.startswith("test_") and row[field] and row[field] not in markdown:
                markdown_missing.append(f"{experiment_id}:{field}")
        for field in ["mse", "mae"]:
            if row[field] and row[field] not in markdown:
                markdown_missing.append(f"{experiment_id}:{field}")
    checks["main_table_markdown_consistent"] = not markdown_missing
    details["main_table_markdown_missing"] = sorted(set(markdown_missing))

    loss_rows = load_csv(builder.LOSS_REGISTRY)
    loss_by_id = {
        row["experiment_id"]: row
        for row in loss_rows
        if row["metric_domain"] == "object_detection"
    }
    loss_failures: list[dict[str, Any]] = []
    for item in builder.EXPERIMENTS[:-1]:
        experiment_id = item["report_id"]
        row = loss_by_id.get(experiment_id)
        if row is None:
            loss_failures.append({"experiment_id": experiment_id, "error": "missing"})
            continue
        if row["all_epochs_finite"] != "True":
            loss_failures.append(
                {"experiment_id": experiment_id, "error": "non-finite loss"}
            )
        results_rows = load_csv(as_path(item["results_csv"]))
        normalized = [
            {str(key).strip(): str(value).strip() for key, value in raw.items()}
            for raw in results_rows
        ]
        best_epoch = int(float(metric_rows[experiment_id]["best_epoch"]))
        best_row = next(
            raw for raw in normalized if int(float(raw["epoch"])) == best_epoch
        )
        for column in builder.LOSS_COLUMNS:
            if row[column] != best_row[column]:
                loss_failures.append(
                    {
                        "experiment_id": experiment_id,
                        "column": column,
                        "registry": row[column],
                        "source": best_row[column],
                    }
                )
    checks["training_validation_losses_separated"] = not loss_failures
    details["loss_failures"] = loss_failures

    test_loss_failures: list[dict[str, Any]] = []
    for item in builder.EXPERIMENTS[:-1]:
        payload = load_json(as_path(item["test_summary"]))
        metrics = payload.get("metrics_yolov10", {})
        loss_keys = [
            key
            for key in metrics
            if str(key).endswith(("box_loss", "cls_loss", "dfl_loss"))
        ]
        log_text = as_path(item["test_log"]).read_text(
            encoding="utf-8", errors="replace"
        )
        registry_row = loss_by_id[item["report_id"]]
        if loss_keys:
            test_loss_failures.append(
                {"experiment_id": item["report_id"], "loss_keys": loss_keys}
            )
        if registry_row["test_loss_status"] != "not_generated_by_test_eval":
            test_loss_failures.append(
                {
                    "experiment_id": item["report_id"],
                    "test_loss_status": registry_row["test_loss_status"],
                }
            )
        if registry_row["test_mse"] or registry_row["test_mae"]:
            test_loss_failures.append(
                {"experiment_id": item["report_id"], "error": "unexpected test loss"}
            )
        if "0.0ms loss" not in log_text:
            test_loss_failures.append(
                {"experiment_id": item["report_id"], "error": "timing marker missing"}
            )
    sys_loss = next(
        row for row in loss_rows if row["experiment_id"] == "SYS5S-001"
    )
    sys_log = as_path(sys_loss["test_log_path"]).read_text(
        encoding="utf-8", errors="replace"
    )
    if (
        sys_loss["test_loss_status"]
        != "full_test_checkpoint_verification_mse_mae"
        or sys_loss["test_mse"] != f"{float(verification['mse']):.12f}"
        or sys_loss["test_mae"] != f"{float(verification['mae']):.12f}"
        or "full test: mse=" not in sys_log
    ):
        test_loss_failures.append(
            {"experiment_id": "SYS5S-001", "error": "flow metric boundary mismatch"}
        )
    checks["test_loss_not_fabricated_or_mixed"] = not test_loss_failures
    details["test_loss_failures"] = test_loss_failures

    percent_rows = load_csv(builder.PERCENT_REGISTRY)
    percent_by_id = {row["metric_id"]: row for row in percent_rows}
    required_percent_ids = {
        "DIAG_TEST_SMALL_GT_SHARE",
        "DIAG_TEST_SMALL_MISS_SHARE",
        "DIAG_VAL_SMALL_GT_SHARE",
        "DIAG_VAL_SMALL_MISS_SHARE",
        "P2_SMALL_AP_GAIN_PP_S2026",
        "P2_SMALL_AP_GAIN_PP_S2027",
        "P2_SMALL_AP_MEAN_GAIN_PP_N2",
        "P2_LATENCY_RATIO_S2026",
        "SYS5S_REL_MEAN_ABS_PREDICTION_DELTA",
    }
    percent_failures: list[dict[str, Any]] = []
    if not required_percent_ids.issubset(percent_by_id):
        percent_failures.append(
            {
                "missing": sorted(required_percent_ids - set(percent_by_id))
            }
        )
    for row in percent_rows:
        if not row["numerator"] or not row["denominator"]:
            percent_failures.append(
                {"metric_id": row["metric_id"], "error": "missing numerator/denominator"}
            )
            continue
        if not row["comparison_object"] or not as_path(row["source_path"]).is_file():
            percent_failures.append(
                {"metric_id": row["metric_id"], "error": "comparison/source missing"}
            )
            continue
        numerator = float(row["numerator"])
        denominator = float(row["denominator"])
        expected = numerator / denominator
        if not math.isclose(
            expected,
            float(row["value_fraction"]),
            rel_tol=0,
            abs_tol=1e-10,
        ):
            percent_failures.append(
                {
                    "metric_id": row["metric_id"],
                    "error": "ratio mismatch",
                    "expected": expected,
                    "value": row["value_fraction"],
                }
            )
    checks["percentage_denominators_complete"] = not percent_failures
    details["percentage_failures"] = percent_failures

    figure_rows = load_csv(builder.FIGURE_MANIFEST)
    figure_failures: list[dict[str, Any]] = []
    if len(figure_rows) != 4:
        figure_failures.append({"error": "expected four figures", "count": len(figure_rows)})
    for row in figure_rows:
        path = as_path(row["path"])
        if not path.is_file():
            figure_failures.append({"figure_id": row["figure_id"], "error": "missing"})
            continue
        actual_hash = builder.sha256_file(path, show_progress=False)
        if actual_hash != row["sha256"]:
            figure_failures.append(
                {"figure_id": row["figure_id"], "error": "manifest hash mismatch"}
            )
    with tempfile.TemporaryDirectory(prefix="stage6_figure_regen_") as temp_dir:
        temp_root = Path(temp_dir)
        regenerated = builder.generate_figures(temp_root)
        regenerated_map = {row["filename"]: row for row in regenerated}
        for row in figure_rows:
            filename = row["filename"]
            if filename not in regenerated_map:
                figure_failures.append(
                    {"figure_id": row["figure_id"], "error": "regeneration missing"}
                )
                continue
            original = np.asarray(Image.open(as_path(row["path"])).convert("RGB"))
            rendered = np.asarray(
                Image.open(temp_root / filename).convert("RGB")
            )
            if original.shape != rendered.shape or not np.array_equal(original, rendered):
                figure_failures.append(
                    {"figure_id": row["figure_id"], "error": "pixel mismatch"}
                )
    checks["main_figures_regenerable"] = not figure_failures
    details["figure_failures"] = figure_failures

    manifest_rows = load_csv(builder.HASH_MANIFEST)
    required_categories = {
        "data",
        "weights",
        "script",
        "config",
        "log_summary",
        "prediction",
        "evidence",
        "metadata",
        "stage6_output",
    }
    manifest_categories = {row["category"] for row in manifest_rows}
    manifest_shape_ok = (
        len(manifest_rows) >= 100
        and required_categories.issubset(manifest_categories)
        and all(row["status"] == "present" for row in manifest_rows)
        and all(len(row["sha256"]) == 64 for row in manifest_rows)
        and all(row["hash_algorithm"] == "SHA-256" for row in manifest_rows)
    )
    checks["sha256_manifest_complete"] = manifest_shape_ok
    details["hash_manifest"] = {
        "entries": len(manifest_rows),
        "categories": sorted(manifest_categories),
        "missing_categories": sorted(required_categories - manifest_categories),
    }

    hash_failures: list[dict[str, str]] = []
    for row in manifest_rows:
        path = as_path(row["path"])
        if not path.exists():
            hash_failures.append({"path": row["path"], "error": "missing"})
            continue
        if row["asset_type"] == "directory":
            digest, file_count, size = builder.sha256_directory(path)
            if (
                digest != row["sha256"]
                or str(file_count) != row["file_count"]
                or str(size) != row["size_bytes"]
            ):
                hash_failures.append(
                    {"path": row["path"], "error": "directory digest mismatch"}
                )
        else:
            digest = builder.sha256_file(path, show_progress=False)
            if digest != row["sha256"] or str(path.stat().st_size) != row["size_bytes"]:
                hash_failures.append(
                    {"path": row["path"], "error": "file digest/size mismatch"}
                )
    checks["sha256_manifest_matches_files"] = not hash_failures
    details["hash_failures"] = hash_failures[:50]
    details["hash_failure_count"] = len(hash_failures)

    old_audit_paths = [
        PROJECT_ROOT / "01_立项" / "阶段2_证据" / "阶段2最终审计.json",
        PROJECT_ROOT / "01_立项" / "阶段3_证据" / "阶段3审计.json",
        PROJECT_ROOT / "01_立项" / "阶段4_证据" / "阶段4审计.json",
        PROJECT_ROOT / "01_立项" / "阶段5_证据" / "阶段5审计.json",
        PROJECT_ROOT / "01_立项" / "阶段5S_证据" / "阶段5S审计.json",
    ]
    old_hashes: dict[str, str] = {}
    for audit_file in old_audit_paths:
        payload = load_json(audit_file)
        for node in walk_dicts(payload):
            path_value = node.get("path")
            hash_value = node.get("sha256")
            if path_value and hash_value:
                old_hashes[normalized_path(path_value)] = str(hash_value).upper()
    manifest_hashes = {
        normalized_path(row["path"]): row["sha256"] for row in manifest_rows
    }
    common_paths = set(old_hashes) & set(manifest_hashes)
    historical_mismatches: list[dict[str, Any]] = []
    eol_normalized_matches = 0
    for path in sorted(common_paths):
        old_hash = old_hashes[path]
        current_hash = manifest_hashes[path]
        if old_hash == current_hash:
            continue
        variant_hash = crlf_variant_sha256(Path(path))
        if variant_hash == old_hash:
            eol_normalized_matches += 1
            continue
        historical_mismatches.append(
            {
                "path": path,
                "old_sha256": old_hash,
                "current_sha256": current_hash,
            }
        )
    checks["historical_evidence_hashes_unchanged"] = (
        len(common_paths) >= 20 and not historical_mismatches
    )
    details["historical_evidence"] = {
        "matched_paths": len(common_paths),
        "eol_normalized_matches": eol_normalized_matches,
        "mismatches": historical_mismatches,
    }

    version_rules = builder.VERSION_RULES.read_text(encoding="utf-8")
    version_tokens = [
        "stage6-evidence-v1.0.0",
        "主版本",
        "次版本",
        "修订号",
        "数字变更登记.csv",
        "旧标签",
    ]
    missing_tokens = [token for token in version_tokens if token not in version_rules]
    checks["version_rules_present"] = not missing_tokens
    details["version_rule_missing_tokens"] = missing_tokens

    metadata = load_json(builder.FREEZE_METADATA)
    boundary_ok = (
        final_map.get("BASELINE-001", {}).get("role")
        == "最终主模型/主结果"
        and metadata.get("main_model") == "BASELINE-001"
        and "不构成端到端性能" in markdown
        and metadata.get("comparison_boundary", "").find("端到端性能") >= 0
        and metadata.get("test_policy", {}).get("stage6_retraining") is False
        and metadata.get("test_policy", {}).get("stage6_test_inference") is False
        and metadata.get("test_policy", {}).get("stage5s_reevaluation") is False
    )
    checks["main_model_and_boundary_declared"] = boundary_ok
    details["main_model_boundary"] = {
        "main_model": metadata.get("main_model"),
        "test_policy": metadata.get("test_policy"),
    }

    summary_ok = False
    summary_text = ""
    if SUMMARY_PATH.is_file():
        summary_text = SUMMARY_PATH.read_text(encoding="utf-8")
        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", summary_text))
        expected_sections = [
            "# 阶段 6 总结",
            "## 一、阶段目标与边界",
            "## 九、阶段验收清单",
            "## 十二、下一阶段唯一启动动作",
        ]
        summary_ok = (
            cjk_count >= 800
            and all(section in summary_text for section in expected_sections)
            and "不开始阶段7" in summary_text
            and "5S" in summary_text
        )
    checks["stage6_summary_present"] = summary_ok
    details["summary"] = {
        "path": str(SUMMARY_PATH),
        "exists": SUMMARY_PATH.is_file(),
        "size_bytes": SUMMARY_PATH.stat().st_size if SUMMARY_PATH.is_file() else 0,
    }

    stage7_summary = (
        PROJECT_ROOT / "01_立项" / "阶段总结" / "阶段7_总结.md"
    )
    checks["stage7_not_started"] = not stage7_summary.exists()
    details["stage7_summary_exists"] = stage7_summary.exists()

    builder_source = required_paths["builder"].read_text(encoding="utf-8")
    forbidden_generator_tokens = [
        "import torch",
        "from ultralytics",
        "run_offline_cascade(",
        "model.val(",
        "model.train(",
    ]
    generator_findings = [
        token for token in forbidden_generator_tokens if token in builder_source
    ]
    checks["generator_does_not_train_or_rerun_inference"] = not generator_findings
    details["generator_forbidden_tokens"] = generator_findings

    artifact_hash_paths = [
        builder.FINAL_TABLE_CSV,
        builder.FINAL_TABLE_MD,
        builder.EXPERIMENT_INDEX,
        builder.LOSS_REGISTRY,
        builder.PERCENT_REGISTRY,
        builder.FIGURE_MANIFEST,
        builder.CHANGE_REGISTER,
        builder.HASH_MANIFEST,
        builder.FREEZE_METADATA,
        builder.VERSION_RULES,
        required_paths["evidence_readme"],
        required_paths["builder"],
        PROJECT_ROOT / "02_代码" / "阶段6" / "audit_stage6.py",
        SUMMARY_PATH,
    ]
    artifact_hashes: dict[str, str] = {}
    artifact_hash_failures: list[str] = []
    for path in artifact_hash_paths:
        if not path.is_file():
            artifact_hash_failures.append(str(path))
            continue
        artifact_hashes[path.name] = builder.sha256_file(
            path, show_progress=False
        )
    checks["stage6_artifact_hashes_recorded"] = not artifact_hash_failures
    details["stage6_artifact_hashes"] = artifact_hashes
    details["stage6_artifact_hash_failures"] = artifact_hash_failures

    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    head = git("rev-parse", "HEAD")
    details["repository_state"] = {
        "branch": branch,
        "head": head,
        "stage5s_tag_commit": stage5s_tag_commit,
        "working_tree_note": "审计在阶段6提交前运行；最终提交和标签由后续Git步骤验证。",
    }
    details["generated_at"] = now_iso()
    details["stage"] = "6"
    details["freeze_version"] = builder.FREEZE_VERSION
    details["checks"] = checks
    details["passed"] = sum(checks.values())
    details["total"] = len(checks)
    details["overall_passed"] = all(checks.values())
    details["manifest_entries"] = len(manifest_rows)
    details["figure_count"] = len(figure_rows)
    details["experiment_count"] = len(final_rows)
    details["detection_experiment_count"] = len(metric_rows)
    details["stage5s_scope"] = (
        "只冻结既有模块级离线联调原型与审计，不重新评估、重做或扩展。"
    )
    details["no_retraining_or_test_inference"] = checks[
        "generator_does_not_train_or_rerun_inference"
    ]
    return checks, details


def main() -> int:
    checks, details = run_audit()
    with AUDIT_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(details, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print(f"audit passed: {details['passed']}/{details['total']}")
    print(f"audit file: {AUDIT_PATH}")
    return 0 if details["overall_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
