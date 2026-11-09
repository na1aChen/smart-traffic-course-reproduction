"""Generate the Chinese Stage 2 Markdown summary from locked artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
EXP_ROOT = PROJECT_ROOT / "03_实验" / "baseline_yolov10n"
TRAIN_DIR = EXP_ROOT / "train"
EVAL_DIR = EXP_ROOT / "test_eval"
EVIDENCE_ROOT = PROJECT_ROOT / "01_立项" / "阶段2_证据"
SUMMARY_ROOT = PROJECT_ROOT / "01_立项" / "阶段总结"
SUMMARY_PATH = SUMMARY_ROOT / "阶段2_总结.md"
AUDIT_PATH = EVIDENCE_ROOT / "阶段2最终审计.json"
TRAIN_SUMMARY_PATH = EVIDENCE_ROOT / "stage2_training_summary.json"
TRAIN_PARTIAL_INVALID_PATH = (
    EVIDENCE_ROOT / "stage2_training_summary.partial_invalid.json"
)
TRAIN_RECOVERED_PATH = EVIDENCE_ROOT / "stage2_training_summary.recovered.json"
EVAL_SUMMARY_PATH = EVIDENCE_ROOT / "stage2_test_evaluation.json"
EVAL_INITIAL_FAILURE_PATH = (
    EVIDENCE_ROOT / "stage2_test_evaluation.initial_failure.json"
)
EVAL_RECOVERED_PATH = EVIDENCE_ROOT / "stage2_test_evaluation.recovered.json"
PROBE_SUMMARY_PATH = EVIDENCE_ROOT / "stage2_probe_summary.json"
COMMAND_COMPARISON_PATH = SCRIPT_DIR / "官方命令与本机命令对照.md"
ENVIRONMENT_SUPPLEMENT_PATH = EVIDENCE_ROOT / "环境补充记录.md"
RECOVERY_RECORD_PATH = EVIDENCE_ROOT / "恢复与失败记录.md"
VISUAL_QA_PATH = EVIDENCE_ROOT / "图表目视QA.md"
EXPERIMENT_TABLE_PATH = PROJECT_ROOT / "03_实验" / "实验总表.csv"
FINAL_STATUS_PATH = (
    PROJECT_ROOT / "03_实验" / "logs" / "BASELINE-001" / "final_status.json"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV while trimming padding from Ultralytics-style headers."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            headers = [header.strip() for header in next(reader)]
        except StopIteration:
            return []
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


def link(path: Path, label: str | None = None) -> str:
    label = label or path.name
    target = path.resolve().as_posix()
    return f"[{label}](/{target})"


def fmt_metric(value: Any) -> str:
    return "不可用" if value is None else f"{float(value):.6f}"


def fmt_mib(value: Any) -> str:
    return "不可用" if value is None else f"{float(value) / (1024**2):.3f}"


def main() -> int:
    required = [
        AUDIT_PATH,
        TRAIN_SUMMARY_PATH,
        EVAL_SUMMARY_PATH,
        PROBE_SUMMARY_PATH,
        TRAIN_DIR / "results.csv",
        TRAIN_DIR / "args.yaml",
        EVAL_DIR / "per_class_metrics.csv",
        COMMAND_COMPARISON_PATH,
        ENVIRONMENT_SUPPLEMENT_PATH,
        RECOVERY_RECORD_PATH,
        VISUAL_QA_PATH,
        EXPERIMENT_TABLE_PATH,
        FINAL_STATUS_PATH,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Stage 2 evidence: {missing}")

    audit = read_json(AUDIT_PATH)
    training = read_json(TRAIN_SUMMARY_PATH)
    evaluation = read_json(EVAL_SUMMARY_PATH)
    probe = read_json(PROBE_SUMMARY_PATH)

    rows = read_csv_rows(TRAIN_DIR / "results.csv")
    for row in rows:
        row["fitness"] = str(
            0.1 * float(row["metrics/mAP50(B)"])
            + 0.9 * float(row["metrics/mAP50-95(B)"])
        )
    best_row = max(rows, key=lambda row: float(row["fitness"]))

    per_class_rows = read_csv_rows(EVAL_DIR / "per_class_metrics.csv")

    status = "通过" if audit.get("status") == "passed" else "未通过"
    summary_metrics = audit.get("summary_metrics", {})
    recomputed = evaluation.get("recomputed_metrics", {})
    yolo_metrics = evaluation.get("metrics_yolov10", {})
    weight_equivalence_path = EVIDENCE_ROOT / "权重等价性校验.json"
    roundtrip_path = (
        EXP_ROOT / "qa_coco_roundtrip" / "qa_coco_roundtrip.json"
    )
    status_path = PROJECT_ROOT / "03_实验" / "logs" / "BASELINE-001" / "postprocess_status.json"
    logs_stdout = PROJECT_ROOT / "03_实验" / "logs" / "BASELINE-001" / "train_stdout.log"
    crash_record = (
        PROJECT_ROOT / "01_立项" / "事件记录" / "Codex会话中断_20261107.md"
    )

    check_labels = {
        "required_artifacts_exist": "训练、评估和审计关键产物齐全",
        "training_status_passed": "正式训练状态为 passed",
        "evaluation_status_passed": "最终测试评估状态为 passed",
        "weights_exactly_equivalent": "权威 HF 权重与训练载体张量完全一致",
        "coco_roundtrip_passed": "YOLO txt → COCO 指标往返测试通过",
        "data_entry_probe_passed": "真实数据 smoke/probe 通过",
        "baseline_config_locked": "seed、输入尺寸、batch、优化器等核心配置已锁定",
        "results_csv_present_and_finite": "训练结果 CSV 存在且数值有限",
        "validation_best_recorded": "验证集最佳轮次和 fitness 可追溯",
        "best_checkpoint_loads": "best.pt 可从 CPU 重新加载",
        "best_checkpoint_matches_csv": "best.pt 的 fitness 与 results.csv 一致",
        "splits_disjoint": "train、val、test 的图像 ID 和内容哈希无重叠",
        "test_ground_truth_complete": "测试集真实值完整",
        "raw_predictions_saved": "测试集原始预测已保存",
        "per_class_metrics_present": "每类指标表已生成",
        "independent_metric_recomputed": "COCO 独立复算指标已生成",
        "training_curves_and_validation_plots_present": "训练曲线和验证图齐全",
    }
    audit_rows = []
    for key, result in audit.get("checks", {}).items():
        mark = "通过" if result.get("passed") else "未通过"
        audit_rows.append(f"| {check_labels.get(key, key)} | {mark} |")

    command_args = training.get("command_arguments", {})
    command_text = (
        "stage2_runner.py train "
        f"--epochs {command_args.get('epochs')} "
        f"--batch {command_args.get('batch')} "
        f"--imgsz {command_args.get('imgsz')} "
        f"--optimizer {command_args.get('optimizer')} "
        f"--patience {command_args.get('patience')} "
        f"--close-mosaic {command_args.get('close_mosaic')} "
        f"--save-period {command_args.get('save_period')} "
        f"--device {command_args.get('device')} "
        f"--workers {command_args.get('workers')} "
        f"--seed {command_args.get('seed')}"
    )

    acceptance_rows = [
        "| 官方命令与本机命令逐项对照 | 通过 |",
        "| 完成至少一个短 smoke test | 通过 |",
        "| 基线使用固定 seed 和保存的划分 | 通过 |",
        "| 训练、验证、测试集职责没有混用 | 通过 |",
        "| 日志、验证曲线、checkpoint 和原始预测齐全 | 通过 |",
        "| 基线总体指标可由原始输出复算 | 通过 |",
        "| 与论文值的差异有配置、预处理或预算层面的解释 | 通过 |",
        "| 阶段总结 `阶段2_总结.md` 已生成 | 通过 |",
    ]
    if status != "通过":
        acceptance_rows = [
            row.replace("| 通过 |", "| 未通过 |")
            for row in acceptance_rows
        ]

    environment_note = (
        "训练启动时缺少 `seaborn`，最初导致标签相关图和混淆矩阵被上游代码跳过；"
        "已在训练期间以 `--no-deps` 安装 `seaborn==0.13.2`，未改变 Torch、NumPy 或模型依赖，"
        "安装后 `pip check` 通过。"
    )
    completed_at = datetime.now().astimezone().isoformat(timespec="seconds")
    reported_gpu_gib = training.get("reported_max_gpu_mem_gib")
    gpu_note = (
        f"训练进度条记录的最高显存约为 {float(reported_gpu_gib):.2f} GiB；"
        "精确的 PyTorch allocated/reserved 回调值因训练汇总恢复不可用。"
        if reported_gpu_gib is not None
        else "训练期的精确 PyTorch allocated/reserved 回调值不可用。"
    )

    lines = [
        "# 阶段 2 总结",
        "",
        "| 字段 | 内容 |",
        "| --- | --- |",
        "| 阶段 | 2，基线复现 |",
        "| 日期 | 2026-11-08 |",
        "| 执行人 | 本人；Codex 辅助执行与复核 |",
        f"| 状态 | {status} |",
        "| 下一阶段 | 阶段 3，错误诊断 |",
        "",
        "## 一、阶段目标与边界",
        "",
        "目标是在锁定的 YOLOv10-N 结构、官方 COCO 权重和 TT100K 45 类固定划分上，建立一个训练、验证、测试职责清楚且指标可复算的基线。",
        "",
        "本阶段没有进行类别加权、结构替换、剪枝、量化、蒸馏或数据增强改造；没有用测试集调参，也没有把 smoke/probe 的短训练结果当作正式基线。",
        "",
        "## 二、完成内容",
        "",
        "### 2.1 数据与权重",
        "",
        "- 训练使用 5,489 张图像和 13,923 个标注框；验证使用 610 张和 1,553 个框；测试使用 3,071 张和 7,706 个框。",
        "- 45 类固定划分保持阶段 1 锁版不变。",
        "- 权威 Hugging Face 权重与可训练 `.pt` 载体完成 595 个张量逐项等价校验，训练构图日志显示 `Transferred 493/595 items from pretrained weights`。",
        "- 测试集只用于验证集选定 `best.pt` 后的唯一一次最终评估。",
        "",
        "### 2.2 训练配置",
        "",
        f"- 实际命令：`{command_text}`。",
        "- 输入尺寸 640，batch 8，最大 40 轮，SGD，初始学习率 0.01，AMP，seed 2026，最后 5 轮关闭 Mosaic，验证集 patience 10。",
        f"- 实际记录轮数：{len(rows)}；最佳轮次：{int(best_row['epoch'])}。",
        f"- 最佳验证 fitness：{fmt_metric(best_row['fitness'])}；验证 mAP50：{fmt_metric(best_row['metrics/mAP50(B)'])}；验证 mAP50-95：{fmt_metric(best_row['metrics/mAP50-95(B)'])}。",
        f"- 正式训练记录耗时：{float(training.get('elapsed_seconds', 0.0)):.3f} 秒（{float(training.get('training_duration_hours_reported', 0.0)):.3f} 小时）。",
        f"- {gpu_note}",
        "",
        "### 2.3 测试集结果",
        "",
        "| 指标 | YOLOv10 内置评估 | COCO 从原始预测复算 |",
        "| --- | ---: | ---: |",
        f"| mAP50 | {fmt_metric(yolo_metrics.get('metrics/mAP50(B)'))} | {fmt_metric(recomputed.get('map50'))} |",
        f"| mAP50-95 | {fmt_metric(yolo_metrics.get('metrics/mAP50-95(B)'))} | {fmt_metric(recomputed.get('map50_95'))} |",
        f"| mAP75 | 不可用 | {fmt_metric(recomputed.get('map75'))} |",
        f"| mAP small | 不可用 | {fmt_metric(recomputed.get('map_small'))} |",
        f"| mAP medium | 不可用 | {fmt_metric(recomputed.get('map_medium'))} |",
        f"| mAP large | 不可用 | {fmt_metric(recomputed.get('map_large'))} |",
        f"| AR100 | 不可用 | {fmt_metric(recomputed.get('mar100'))} |",
        "",
        f"- 测试集图像数：{recomputed.get('images')}；类别数：{recomputed.get('categories')}。",
        f"- 每类指标表记录类别数：{len(per_class_rows)}。",
        f"- 推理速度（YOLO 验证器，毫秒/图）：{fmt_metric(yolo_metrics.get('inference_ms_per_image'))}。",
        "",
        "## 三、新增或修改文件",
        "",
        f"- 阶段总结：`{SUMMARY_PATH.resolve()}`",
        f"- 训练汇总：{link(TRAIN_SUMMARY_PATH)}",
        f"- 训练汇总损坏前缀：{link(TRAIN_PARTIAL_INVALID_PATH, 'stage2_training_summary.partial_invalid.json')}",
        f"- 训练汇总恢复件：{link(TRAIN_RECOVERED_PATH, 'stage2_training_summary.recovered.json')}",
        f"- 测试汇总：{link(EVAL_SUMMARY_PATH)}",
        f"- 首次失败的测试汇总：{link(EVAL_INITIAL_FAILURE_PATH, 'stage2_test_evaluation.initial_failure.json')}",
        f"- 测试评估恢复件：{link(EVAL_RECOVERED_PATH, 'stage2_test_evaluation.recovered.json')}",
        f"- 最终审计：{link(AUDIT_PATH)}",
        f"- 实验总表：{link(EXPERIMENT_TABLE_PATH)}",
        f"- 恢复后最终运行状态：{link(FINAL_STATUS_PATH)}",
        f"- 权重等价性：{link(weight_equivalence_path)}",
        f"- COCO 往返测试：{link(roundtrip_path)}",
        f"- 训练结果表：{link(TRAIN_DIR / 'results.csv')}",
        f"- 每类指标：{link(EVAL_DIR / 'per_class_metrics.csv')}",
        f"- 训练日志：{link(logs_stdout)}",
        f"- 评估原始预测：{link(EVAL_DIR / 'predictions_category0.json')}",
        f"- 官方与本机命令对照：{link(COMMAND_COMPARISON_PATH)}",
        f"- 环境补充记录：{link(ENVIRONMENT_SUPPLEMENT_PATH)}",
        f"- 恢复与失败记录：{link(RECOVERY_RECORD_PATH)}",
        f"- 图表目视 QA：{link(VISUAL_QA_PATH)}",
        f"- 首次中断记录：{link(crash_record)}",
        "",
        "## 四、关键命令、配置与校验值",
        "",
        "```text",
        "02_代码\\环境\\venv\\Scripts\\python.exe 02_代码\\阶段2\\stage2_runner.py train "
        "--epochs 40 --batch 8 --imgsz 640 --optimizer SGD --patience 10 "
        "--close-mosaic 5 --save-period 5 --device 0 --workers 0 --seed 2026",
        "02_代码\\环境\\venv\\Scripts\\python.exe 02_代码\\阶段2\\stage2_runner.py evaluate "
        "--batch 8 --imgsz 640 --device 0 --workers 0 --seed 2026",
        "```",
        "",
        "关键文件 SHA-256：",
        "",
        f"- `stage2_runner.py`：`{sha256(SCRIPT_DIR / 'stage2_runner.py')}`",
        f"- `results.csv`：`{sha256(TRAIN_DIR / 'results.csv')}`",
        f"- `best.pt`：`{sha256(TRAIN_DIR / 'weights' / 'best.pt')}`",
        f"- `predictions_category0.json`：`{sha256(EVAL_DIR / 'predictions_category0.json')}`",
        "",
        "## 五、验收清单逐项结果",
        "",
        "| 验收项 | 结果 |",
        "| --- | --- |",
        *acceptance_rows,
        "",
        "最终审计子项：",
        "",
        "| 检查项 | 结果 |",
        "| --- | --- |",
        *audit_rows,
        "",
        "## 六、事实、推断和建议",
        "",
        "### 已确认事实",
        "",
        f"- 正式训练状态为 `{training.get('status')}`，最终测试评估状态为 `{evaluation.get('status')}`，最终审计状态为 `{audit.get('status')}`。",
        "- 训练、验证和测试使用同一份固定 45 类数据清单；split 检查未发现图像 ID 或内容哈希重叠。",
        f"- 最佳轮次为 {int(best_row['epoch'])}，测试集主指标由已保存原始预测独立复算。",
        "",
        "### 推断与解释",
        "",
        "- 官方论文报告的 YOLOv10-N 38.5% COCO AP 是 80 类 COCO 验证集结果；本实验是 45 类 TT100K 单卡迁移训练，类别、数据分布、训练轮数和 batch 均不同，两组数值不能直接做等号比较。",
        "- 当前只完成一个固定 seed 的基线，不据此声称结果稳定或具有统计显著性。",
        "- 每类指标、混淆矩阵和小/中/大目标 AP 的差异将在阶段 3 诊断，本阶段不提前归因于某一种改进机制。",
        "",
        "### 建议",
        "",
        "- 下一阶段先从每类 AP、目标尺寸分桶和典型漏检/误检证据中寻找一个可测的失败机制。",
        "- 不因模型名或单一总体 mAP 直接认定改进有效；改进实验必须与 `BASELINE-001` 使用相同训练预算和划分。",
        "",
        "## 七、失败、异常与未解决问题",
        "",
        f"- 2026-11-07 的 Codex 主进程崩溃曾中断阶段 2 的早期 smoke test；该次未完成运行不计入正式基线，原始记录见 {link(crash_record, 'Codex 会话中断记录')}。",
        f"- {environment_note}",
        f"- 40 轮训练和全部图表正常完成后，训练运行器的 JSON 汇总因 `argparse vars(args)` 中包含不可序列化的 `func` 函数对象而失败；完整产物保存在 `{TRAIN_DIR.resolve()}`，恢复过程见 {link(TRAIN_RECOVERED_PATH, '训练汇总恢复件')}，损坏前缀见 {link(TRAIN_PARTIAL_INVALID_PATH, '训练汇总损坏前缀')}。",
        f"- 唯一一次测试前向正常完成并保存了 3,071 个原始预测文件，但每类指标汇总把整数键名称字典按字符串键访问，触发 `KeyError: '0'`。修复后只使用已保存预测做离线 COCO 复算，没有重新加载图像或重跑测试推理；首次失败记录见 {link(EVAL_INITIAL_FAILURE_PATH, '首次失败测试汇总')}，恢复记录见 {link(EVAL_RECOVERED_PATH, '测试评估恢复件')}。",
        "- Ultralytics 的 `strip_optimizer` 会把最终 `best.pt` 中的 epoch 字段重置为 `-1`；最佳轮次身份通过 `train_metrics.fitness` 与 `results.csv` 第 39 轮 fitness 一致来核验，不能把 `-1` 写进训练轨迹。",
        "- 当前未做多 seed，因此本阶段结果只能作为单次基线，不能给出标准差。",
        "- 当前未进行模型导出延迟基准；YOLO PyTorch 推理速度不能代替论文的导出后端延迟。",
        "- 报告、PPT、阶段 3 诊断和最终证据冻结均未在本阶段执行。",
        "",
        "## 八、对下一阶段的影响",
        "",
        "- 阶段 3 应以当前 `best.pt` 和冻结的测试原始预测为诊断对象，不重新挑选测试集指标。",
        "- 任何改进实验都必须新建实验 ID，不能覆盖本阶段配置、日志或 checkpoint。",
        "- 若后续要比较改进效果，训练轮数、batch、输入尺寸、优化器和划分应保持一致；否则应明确写为组合变化。",
        "",
        "## 九、下一阶段唯一启动动作",
        "",
        "进入阶段 3：读取 `BASELINE-001` 的固定 `best.pt`、每类指标和测试原始预测，完成类别 AP、小/中/大目标分桶、主要混淆对和至少 8 个典型漏检/误检案例诊断；在本阶段总结通过前不进入阶段 4。",
        "",
        f"生成时间：{completed_at}。",
        f"恢复后最终状态：`{FINAL_STATUS_PATH}`。",
        f"恢复后后处理最终状态（仍保留 `evaluation_exit_code=1` 现场）：`{status_path}`。",
        "",
    ]
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Stage 2 summary: {SUMMARY_PATH}")
    print(f"Status: {status}")
    return 0 if status == "通过" else 1


if __name__ == "__main__":
    raise SystemExit(main())
