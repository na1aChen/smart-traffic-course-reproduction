from __future__ import annotations

import copy
import csv
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = PROJECT_ROOT / "04_报告"
WORKSPACE = REPORT_ROOT / "工作区"
FIGURE_ROOT = REPORT_ROOT / "图表"
TEMPLATE_DOCX = (
    REPORT_ROOT / "模板副本" / "智能交通-课程报告_大作业2_骨架.docx"
)
CONTENT_PATH = WORKSPACE / "报告正文.md"
OUTPUT_DOCX = (
    REPORT_ROOT / "基于TT100K的交通标志小目标检测复现与验证性改进报告.docx"
)
BUILD_RECORD = PROJECT_ROOT / "01_立项" / "阶段7_证据" / "报告构建记录.json"
CITATION_MAP = WORKSPACE / "数字引用映射.csv"

TEMPLATE_SHA256 = (
    "FDD2C42B3ACDEF9A412DB2F3CCAA2D85D26A7DD081DA876CF3CB325D5CC00662"
)
REPORT_TITLE = "基于 TT100K 的交通标志小目标检测复现与验证性改进"

FINAL_TABLE = PROJECT_ROOT / "01_立项" / "阶段6_证据" / "最终主表.csv"
PERCENT_REGISTRY = (
    PROJECT_ROOT / "01_立项" / "阶段6_证据" / "百分比口径登记.csv"
)
DATASET_SUMMARY = (
    PROJECT_ROOT
    / "03_实验"
    / "datasets"
    / "TT100K"
    / "yolo"
    / "manifests"
    / "dataset_summary.json"
)
FORECAST_RESULT = (
    PROJECT_ROOT / "01_立项" / "阶段5S_证据" / "forecast_result.json"
)
STAGE5S_ASSET_CHECK = (
    PROJECT_ROOT / "01_立项" / "阶段5S_证据" / "阶段5S_资产检查.json"
)
STAGE5S_AUDIT = (
    PROJECT_ROOT / "01_立项" / "阶段5S_证据" / "阶段5S审计.json"
)
ANALYSIS_SUMMARY = (
    PROJECT_ROOT
    / "03_实验"
    / "诊断"
    / "BASELINE-001"
    / "analysis"
    / "analysis_summary.json"
)

STAGE6_FIGURES = {
    "baseline_losses": (
        PROJECT_ROOT
        / "01_立项"
        / "阶段6_证据"
        / "主图"
        / "fig2_baseline_training_validation_losses.png"
    ),
    "size_bucket_ap": (
        PROJECT_ROOT
        / "01_立项"
        / "阶段6_证据"
        / "主图"
        / "fig3_test_size_bucket_ap.png"
    ),
    "detection_main_results": (
        PROJECT_ROOT
        / "01_立项"
        / "阶段6_证据"
        / "主图"
        / "fig1_detection_main_results.png"
    ),
    "cascade_difference": (
        PROJECT_ROOT
        / "01_立项"
        / "阶段6_证据"
        / "主图"
        / "fig4_sys5s_cascade_difference.png"
    ),
}

FIGURE_SPECS = {
    "research_route": {
        "number": 1,
        "path": FIGURE_ROOT / "图1_研究路线.png",
        "caption": "图1 从证据锁版到模块级联的研究路线",
        "width": Inches(5.62),
        "alt": "研究路线图，依次为数据与来源锁版、基线复现、错误诊断、P2单因子、两seed与预算、5S离线联调",
        "source": "来源：本项目阶段总结、实验总表与阶段6冻结证据。",
    },
    "baseline_losses": {
        "number": 2,
        "path": FIGURE_ROOT / "图2_基线训练与验证损失.png",
        "caption": "图2 BASELINE-001训练与验证损失",
        "width": Inches(5.62),
        "alt": "基线模型四十轮训练中box、cls和DFL损失在训练与验证集上的变化曲线",
        "source": (
            "来源：03_实验/baseline_yolov10n/train/results.csv；"
            "图中不含测试损失。"
        ),
    },
    "size_bucket_ap": {
        "number": 3,
        "path": FIGURE_ROOT / "图3_测试尺寸分桶AP.png",
        "caption": "图3 基线与P2的测试集尺寸分桶AP",
        "width": Inches(5.42),
        "alt": "基线和P2两seed在small、medium、large尺寸桶上的测试mAP50-95均值和散点",
        "source": (
            "来源：01_立项/阶段5_证据/stage5_experiment_metrics.csv；"
            "均值由两个seed计算。"
        ),
    },
    "detection_main_results": {
        "number": 4,
        "path": FIGURE_ROOT / "图4_检测主结果.png",
        "caption": "图4 检测总体AP与小目标AP主结果",
        "width": Inches(5.62),
        "alt": "基线两seed、P2两seed和四十八轮预算对照的总体AP与小目标AP柱状图",
        "source": (
            "来源：01_立项/阶段6_证据/最终主表.csv；"
            "柱为均值，误差线和黑点为两个seed。"
        ),
    },
    "system_dataflow": {
        "number": 5,
        "path": FIGURE_ROOT / "图5_模块级联数据流.png",
        "caption": "图5 检测事件到流量预测的离线模块级联数据流",
        "width": Inches(5.62),
        "alt": "离线模块级联图，从基线检测事件、图像ID顺序适配器、PEMS输入、TA-iTransformer到预测输出",
        "source": (
            "来源：01_立项/阶段5S_证据/阶段5S预注册与接口设计.md；"
            "01_立项/阶段5S_证据/forecast_result.json。"
        ),
    },
    "cascade_difference": {
        "number": 6,
        "path": FIGURE_ROOT / "图6_单窗口预测绝对差.png",
        "caption": "图6 SYS5S-001单窗口预测绝对差",
        "width": Inches(5.45),
        "alt": "单窗口九十六个时间步和一百七十个传感器通道的注入前后预测绝对差热图",
        "source": (
            "来源：01_立项/阶段5S_证据/cascade_predictions.npz；"
            "亮色仅表示输出差异，不表示预测精度。"
        ),
    },
}

TABLE_ORDER = {
    "data_split": 1,
    "training_config": 2,
    "baseline_metrics": 3,
    "p2_single_seed": 4,
    "sys5s_evidence": 5,
}

PRESERVE_ONLY_PARTS = [
    "_rels/.rels",
    "customXml/_rels/item1.xml.rels",
    "customXml/item1.xml",
    "customXml/itemProps1.xml",
    "docProps/app.xml",
    "docProps/core.xml",
    "docProps/custom.xml",
    "word/fontTable.xml",
    "word/footer1.xml",
    "word/header1.xml",
    "word/media/image1.png",
    "word/media/image2.png",
    "word/numbering.xml",
    "word/styles.xml",
    "word/theme/theme1.xml",
]


def now_iso() -> str:
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


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


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
        writer.writerows(rows)


def final_metric_rows() -> dict[str, dict[str, str]]:
    return {row["experiment_id"]: row for row in load_csv(FINAL_TABLE)}


def prepare_report_figures() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    for key, source in STAGE6_FIGURES.items():
        shutil.copy2(source, FIGURE_SPECS[key]["path"])
    missing = [
        str(spec["path"])
        for spec in FIGURE_SPECS.values()
        if not spec["path"].is_file()
    ]
    if missing:
        raise FileNotFoundError("报告图缺失: " + "; ".join(missing))


def parse_content() -> dict[str, list[tuple[str, str]]]:
    text = CONTENT_PATH.read_text(encoding="utf-8")
    sections: dict[str, list[tuple[str, str]]] = {}
    current: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        if current is None:
            buffer = []
            return
        block = "\n".join(buffer).strip()
        if block:
            marker = re.fullmatch(r"\{\{(TABLE|FIGURE):([a-z0-9_]+)\}\}", block)
            if marker:
                sections[current].append((marker.group(1).lower(), marker.group(2)))
            else:
                sections[current].append(("paragraph", block))
        buffer = []

    for line in text.splitlines():
        if line.startswith("## "):
            flush()
            current = line[3:].strip()
            sections[current] = []
        elif current is not None:
            if line.strip() == "":
                flush()
            else:
                buffer.append(line.rstrip())
    flush()
    return sections


def template_heading(title: str) -> str:
    if title == "6 实验总结与收获":
        return "实验总结与收获"
    return title


def set_east_asian_font(run, font_name: str = "宋体", size_pt: float | None = None) -> None:
    run.font.name = "Times New Roman"
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), font_name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), "Times New Roman")
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), "Times New Roman")
    if size_pt is not None:
        run.font.size = Pt(size_pt)


def style_body_paragraph(paragraph) -> None:
    paragraph.style = "Normal"
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    fmt = paragraph.paragraph_format
    fmt.first_line_indent = Pt(21)
    fmt.line_spacing = 1.5
    fmt.space_before = Pt(0)
    fmt.space_after = Pt(0)
    fmt.widow_control = True
    for run in paragraph.runs:
        set_east_asian_font(run, "宋体", 10.5)
        run.font.color.rgb = RGBColor(0, 0, 0)


def style_caption(paragraph, keep_with_next: bool = False) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fmt = paragraph.paragraph_format
    fmt.first_line_indent = None
    fmt.line_spacing = 1.0
    fmt.space_before = Pt(6)
    fmt.space_after = Pt(4)
    fmt.keep_with_next = keep_with_next
    for run in paragraph.runs:
        set_east_asian_font(run, "宋体", 10)
        run.bold = True
        run.font.color.rgb = RGBColor(0, 0, 0)


def style_source_note(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    fmt = paragraph.paragraph_format
    fmt.first_line_indent = None
    fmt.line_spacing = 1.0
    fmt.space_before = Pt(3)
    fmt.space_after = Pt(6)
    for run in paragraph.runs:
        set_east_asian_font(run, "宋体", 9)
        run.font.color.rgb = RGBColor(0x59, 0x59, 0x59)


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shading = tc_pr.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        tc_pr.append(shading)
    shading.set(qn("w:fill"), fill)


def set_cell_margins(table, top: int = 80, start: int = 90, bottom: int = 80, end: int = 90) -> None:
    tbl_pr = table._tbl.tblPr
    margins = tbl_pr.first_child_found_in("w:tblCellMar")
    if margins is None:
        margins = OxmlElement("w:tblCellMar")
        tbl_pr.append(margins)
    for tag, value in [("top", top), ("left", start), ("bottom", bottom), ("right", end)]:
        node = margins.find(qn(f"w:{tag}"))
        if node is None:
            node = OxmlElement(f"w:{tag}")
            margins.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_cell_borders(cell, color: str = "D9D9D9", size: int = 4) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        element = borders.find(qn(f"w:{edge}"))
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), str(size))
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def set_cell_text(cell, text: str, bold: bool = False, color: str = "000000") -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fmt = paragraph.paragraph_format
    fmt.first_line_indent = None
    fmt.line_spacing = 1.0
    fmt.space_before = Pt(0)
    fmt.space_after = Pt(0)
    run = paragraph.add_run(str(text))
    set_east_asian_font(run, "宋体", 9.5)
    run.bold = bold
    run.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_table_after(
    doc,
    anchor,
    number: int,
    caption: str,
    headers: list[str],
    rows: list[list[str]],
    widths: list[float],
    source: str,
):
    caption_p = doc.add_paragraph()
    caption_p.add_run(f"表{number} {caption}")
    style_caption(caption_p, keep_with_next=True)
    anchor.addnext(caption_p._p)
    cursor = caption_p._p

    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for index, width in enumerate(widths):
        table.columns[index].width = Inches(width)
    cursor.addnext(table._tbl)
    cursor = table._tbl

    set_cell_margins(table)
    for row_index, row in enumerate(table.rows):
        if row_index == 0:
            set_repeat_table_header(row)
        for col_index, cell in enumerate(row.cells):
            set_cell_borders(cell)
            value = headers[col_index] if row_index == 0 else rows[row_index - 1][col_index]
            if row_index == 0:
                set_cell_shading(cell, "1F4E79")
                set_cell_text(cell, value, bold=True, color="FFFFFF")
            else:
                set_cell_shading(cell, "F5F8FA" if row_index % 2 == 0 else "FFFFFF")
                set_cell_text(cell, value)
            cell.width = Inches(widths[col_index])
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.keep_with_next = (
                    row_index < len(rows)
                )
        if row_index > 0:
            tr_pr = row._tr.get_or_add_trPr()
            tr_pr.append(OxmlElement("w:cantSplit"))

    source_p = doc.add_paragraph()
    source_p.add_run(source)
    style_source_note(source_p)
    table._tbl.addnext(source_p._p)
    cursor = source_p._p
    return cursor


def set_alt_text(paragraph, alt_text: str) -> None:
    doc_prs = paragraph._p.findall(".//" + qn("wp:docPr"))
    for doc_pr in doc_prs:
        doc_pr.set("descr", alt_text)
        doc_pr.set("title", alt_text)


def add_figure_after(
    doc,
    anchor,
    key: str,
):
    spec = FIGURE_SPECS[key]
    path = spec["path"]
    if not path.is_file():
        raise FileNotFoundError(f"报告图不存在: {path}")

    image_p = doc.add_paragraph()
    image_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    image_p.paragraph_format.first_line_indent = None
    image_p.paragraph_format.line_spacing = 1.0
    image_p.paragraph_format.space_before = Pt(6)
    image_p.paragraph_format.space_after = Pt(2)
    image_p.paragraph_format.keep_with_next = True
    image_p.add_run().add_picture(str(path), width=spec["width"])
    set_alt_text(image_p, spec["alt"])
    anchor.addnext(image_p._p)
    cursor = image_p._p

    caption_p = doc.add_paragraph()
    caption_p.add_run(spec["caption"])
    style_caption(caption_p)
    cursor.addnext(caption_p._p)
    cursor = caption_p._p

    source_p = doc.add_paragraph()
    source_p.add_run(spec["source"])
    style_source_note(source_p)
    cursor.addnext(source_p._p)
    cursor = source_p._p
    return cursor


def table_data() -> dict[str, dict[str, Any]]:
    final_rows = final_metric_rows()
    dataset = load_json(DATASET_SUMMARY)
    forecast = load_json(FORECAST_RESULT)
    asset = load_json(STAGE5S_ASSET_CHECK)
    stage5s_audit = load_json(STAGE5S_AUDIT)
    baseline = final_rows["BASELINE-001"]
    verification = forecast["full_test_checkpoint_verification"]

    data_split = {
        "caption": "TT100K 45类固定数据划分",
        "headers": ["划分", "图像数", "标注框数", "用途"],
        "rows": [
            ["train", f"{dataset['derived_train_images']:,}", f"{dataset['selected_boxes']['train']:,}", "训练与参数更新"],
            ["val", f"{dataset['derived_val_images']:,}", f"{dataset['selected_boxes']['val']:,}", "选择best.pt与诊断确认"],
            ["test", f"{dataset['official_test_images']:,}", f"{dataset['selected_boxes']['test']:,}", "冻结模型最终评估"],
        ],
        "widths": [1.0, 1.2, 1.3, 2.2],
        "source": f"来源：{DATASET_SUMMARY.relative_to(PROJECT_ROOT).as_posix()}。",
    }

    experiment_order = [
        "BASELINE-001",
        "BASELINE-002-S2027",
        "P2-001",
        "P2-002-S2027",
        "BASELINE-BUDGET48-S2026",
    ]
    training_config = {
        "caption": "检测实验的训练配方与最佳验证轮",
        "headers": ["实验ID", "seed", "epochs", "imgsz", "batch", "最佳验证轮"],
        "rows": [
            [
                experiment_id,
                final_rows[experiment_id]["seed"],
                final_rows[experiment_id]["epochs"],
                final_rows[experiment_id]["imgsz"],
                final_rows[experiment_id]["batch"],
                final_rows[experiment_id]["best_epoch"],
            ]
            for experiment_id in experiment_order
        ],
        "widths": [2.10, 0.55, 0.55, 0.55, 0.55, 0.95],
        "source": (
            "来源：01_立项/阶段6_证据/最终主表.csv；"
            "03_实验/configs/。所有实验均使用SGD、AMP、初始学习率0.01。"
        ),
    }

    baseline_metrics = {
        "caption": "BASELINE-001测试指标的内置值与COCO复算值",
        "headers": ["指标", "YOLOv10内置", "COCO复算", "说明"],
        "rows": [
            ["mAP50", "0.501000", baseline["test_map50"], "正文采用复算值"],
            ["mAP50-95", "0.376000", baseline["test_map50_95"], "正文采用复算值"],
            ["small AP", "—", baseline["test_map_small"], "面积小于32²"],
            ["medium AP", "—", baseline["test_map_medium"], "面积32²至96²"],
            ["large AP", "—", baseline["test_map_large"], "面积大于96²"],
            ["AR100", "—", baseline["test_mar100"], "每图最多100框"],
        ],
        "widths": [1.15, 1.20, 1.15, 2.20],
        "source": (
            "来源：01_立项/阶段2_证据/stage2_test_evaluation.json；"
            "01_立项/阶段5_证据/stage5_experiment_metrics.csv。"
        ),
    }

    p2_table = {
        "caption": "基线、P2、第二seed与预算对照的测试结果",
        "headers": ["实验ID", "seed", "mAP50", "mAP50-95", "small AP", "large AP", "ms/图"],
        "rows": [
            [
                experiment_id,
                final_rows[experiment_id]["seed"],
                final_rows[experiment_id]["test_map50"],
                final_rows[experiment_id]["test_map50_95"],
                final_rows[experiment_id]["test_map_small"],
                final_rows[experiment_id]["test_map_large"],
                final_rows[experiment_id]["inference_ms"],
            ]
            for experiment_id in experiment_order
        ],
        "widths": [1.65, 0.42, 0.65, 0.75, 0.68, 0.68, 0.85],
        "source": (
            "来源：01_立项/阶段6_证据/最终主表.csv；"
            "推理时延为同一YOLO验证器口径，不等同于导出后端FPS。"
        ),
    }

    sys5s = {
        "caption": "SYS5S-001接口与复算证据",
        "headers": ["项目", "结果", "证据口径"],
        "rows": [
            ["检测事件 / 唯一图像", "6,780 / 2,621", "BASELINE-001测试预测事件"],
            ["代理分箱", "96", "图像ID顺序等频分箱"],
            ["输入 / 输出形状", "96×170 / 96×170", "PEMS08通道结构"],
            ["平均绝对预测差", "0.017423", "单窗口注入前后"],
            ["完整测试MSE", f"{verification['mse']:.9f}", f"{verification['windows']:,}个窗口"],
            ["完整测试MAE", f"{verification['mae']:.9f}", f"{verification['values']:,}个预测值"],
            [
                "Checkpoint严格加载",
                (
                    f"{asset['checkpoint']['state_entries']}张量 / "
                    f"{asset['checkpoint']['parameter_tensors']:,}元素"
                ),
                "missing 0 / unexpected 0",
            ],
            [
                "运行 / 阶段审计",
                (
                    f"{len(forecast['checks'])}/{len(forecast['checks'])} / "
                    f"{stage5s_audit['passed']}/{stage5s_audit['total']}"
                ),
                "自动检查与冻结审计",
            ],
        ],
        "widths": [1.8, 1.55, 2.35],
        "source": (
            "来源：01_立项/阶段5S_证据/forecast_result.json；"
            "阶段5S_资产检查.json；阶段5S审计.json。"
        ),
    }

    return {
        "data_split": data_split,
        "training_config": training_config,
        "baseline_metrics": baseline_metrics,
        "p2_single_seed": p2_table,
        "sys5s_evidence": sys5s,
    }


def set_cover_title(doc: Document) -> None:
    paragraph = next(
        p for p in doc.paragraphs if p.text.strip().startswith("报告题目")
    )
    if not paragraph.runs:
        raise RuntimeError("封面报告题目段落缺少可编辑run")
    value_run = copy.deepcopy(paragraph.runs[-1]._r)
    paragraph.runs[0].text = "报告题目："
    for run in paragraph.runs[1:]:
        run._r.getparent().remove(run._r)
    for text_node in value_run.findall(".//" + qn("w:t")):
        text_node.text = ""
    text_node = value_run.find(qn("w:t"))
    if text_node is None:
        text_node = OxmlElement("w:t")
        value_run.append(text_node)
    text_node.set(qn("xml:space"), "preserve")
    text_node.text = " " + REPORT_TITLE
    paragraph.runs[0]._r.addnext(value_run)


def mark_template_table_headers(doc: Document) -> None:
    for table in doc.tables:
        if table.rows:
            set_repeat_table_header(table.rows[0])


def prepare_headings(doc: Document, content: dict[str, list[tuple[str, str]]]) -> dict[str, Any]:
    anchors: dict[str, Any] = {}
    for title in content:
        target = template_heading(title)
        paragraph = next(
            (p for p in doc.paragraphs if p.text.strip() == target),
            None,
        )
        if paragraph is None:
            raise KeyError(f"模板缺少目标标题: {target}")
        anchors[title] = paragraph

    chapter_titles = [
        "研究背景与意义",
        "模型介绍",
        "实验训练结果",
        "模型改进与实验",
        "拓展实验内容",
        "实验总结与收获",
    ]
    for chapter_index, chapter_title in enumerate(chapter_titles):
        paragraph = next(p for p in doc.paragraphs if p.text.strip() == chapter_title)
        paragraph.paragraph_format.keep_with_next = True
        paragraph.paragraph_format.space_before = Pt(12)
        paragraph.paragraph_format.space_after = Pt(8)
        if chapter_index > 0:
            paragraph.paragraph_format.page_break_before = False

    for title, anchor in anchors.items():
        if re.match(r"^\d+\.\d+\s", title):
            anchor.paragraph_format.keep_with_next = True
            anchor.paragraph_format.space_before = Pt(8)
            anchor.paragraph_format.space_after = Pt(4)

    first_heading = next(
        p for p in doc.paragraphs if p.text.strip() == "研究背景与意义"
    )
    element = first_heading._p.getnext()
    while element is not None:
        next_element = element.getnext()
        if element.tag == qn("w:p"):
            text = "".join(node.text or "" for node in element.findall(".//" + qn("w:t")))
            has_sensitive = bool(
                element.findall(".//" + qn("w:drawing"))
                or element.findall(".//" + qn("w:sectPr"))
                or element.findall(".//" + qn("w:br"))
                or element.findall(".//" + qn("w:fldChar"))
            )
            if not text.strip() and not has_sensitive:
                element.getparent().remove(element)
        element = next_element
    return anchors


def insert_sections(doc: Document, content: dict[str, list[tuple[str, str]]], anchors: dict[str, Any]) -> dict[str, int]:
    tables = table_data()
    stats = {"paragraphs": 0, "tables": 0, "figures": 0}
    for title, blocks in content.items():
        cursor = anchors[title]._p
        for block_type, value in blocks:
            if block_type == "paragraph":
                paragraph = doc.add_paragraph()
                paragraph.add_run(value)
                style_body_paragraph(paragraph)
                if re.search(r"[A-Fa-f0-9]{32,}|https?://", value):
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
                cursor.addnext(paragraph._p)
                cursor = paragraph._p
                stats["paragraphs"] += 1
            elif block_type == "table":
                spec = tables[value]
                cursor = add_table_after(
                    doc,
                    cursor,
                    TABLE_ORDER[value],
                    spec["caption"],
                    spec["headers"],
                    spec["rows"],
                    spec["widths"],
                    spec["source"],
                )
                stats["tables"] += 1
            elif block_type == "figure":
                cursor = add_figure_after(doc, cursor, value)
                stats["figures"] += 1
            else:
                raise ValueError(f"未知内容块: {block_type}")
    return stats


def add_first_page_blank_header_and_update_fields(input_docx: Path, output_docx: Path) -> dict[str, str]:
    with zipfile.ZipFile(input_docx, "r") as source:
        parts = {name: source.read(name) for name in source.namelist()}

    rels_name = "word/_rels/document.xml.rels"
    rels = ET.fromstring(parts[rels_name])
    rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    ids = [
        rel.attrib["Id"]
        for rel in rels.findall(f"{{{rel_ns}}}Relationship")
    ]
    numeric_ids = [int(value[3:]) for value in ids if re.fullmatch(r"rId\d+", value)]
    header_rid = f"rId{max(numeric_ids or [0]) + 1}"
    ET.SubElement(
        rels,
        f"{{{rel_ns}}}Relationship",
        {
            "Id": header_rid,
            "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/header",
            "Target": "header2.xml",
        },
    )
    ET.register_namespace("", rel_ns)
    parts[rels_name] = ET.tostring(rels, encoding="utf-8", xml_declaration=True)

    parts["word/header2.xml"] = (
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        b'<w:hdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        b"<w:p/></w:hdr>"
    )

    content_types_name = "[Content_Types].xml"
    content_types = ET.fromstring(parts[content_types_name])
    ct_ns = "http://schemas.openxmlformats.org/package/2006/content-types"
    ET.SubElement(
        content_types,
        f"{{{ct_ns}}}Override",
        {
            "PartName": "/word/header2.xml",
            "ContentType": "application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml",
        },
    )
    ET.register_namespace("", ct_ns)
    parts[content_types_name] = ET.tostring(
        content_types, encoding="utf-8", xml_declaration=True
    )

    document_xml = parts["word/document.xml"].decode("utf-8")
    first_header_ref = f'<w:headerReference xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" r:id="{header_rid}" w:type="first"/>'
    document_xml = document_xml.replace("<w:sectPr>", f"<w:sectPr>{first_header_ref}", 1)
    parts["word/document.xml"] = document_xml.encode("utf-8")

    settings_name = "word/settings.xml"
    settings_xml = parts[settings_name].decode("utf-8")
    if "<w:updateFields" not in settings_xml:
        settings_xml = settings_xml.replace(
            "</w:settings>",
            '<w:updateFields w:val="true"/></w:settings>',
        )
        parts[settings_name] = settings_xml.encode("utf-8")

    with zipfile.ZipFile(TEMPLATE_DOCX, "r") as reference:
        reference_names = reference.namelist()
        for name in PRESERVE_ONLY_PARTS:
            parts[name] = reference.read(name)
        for name in reference_names:
            if name.endswith("/") and name not in parts:
                parts[name] = b""

    with zipfile.ZipFile(output_docx, "w", zipfile.ZIP_DEFLATED) as target:
        for name in sorted(parts):
            target.writestr(name, parts[name])
    return {
        "first_page_header_relationship": header_rid,
        "blank_header_part": "word/header2.xml",
        "update_fields": "true",
        "preserve_only_parts_restored": PRESERVE_ONLY_PARTS,
    }


def validate_percentages(content_text: str) -> list[str]:
    allowed = {
        row["report_display"]
        for row in load_csv(PERCENT_REGISTRY)
        if row["report_display"].endswith("%")
    }
    found = re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?%", content_text)
    unknown = sorted({token for token in found if token not in allowed})
    if unknown:
        raise ValueError(f"报告出现未登记百分数: {unknown}")
    return found


def build_citation_map() -> None:
    rows = [
        {
            "report_element": "表1",
            "usage": "TT100K固定划分图像数与标注框数",
            "source_path": DATASET_SUMMARY.relative_to(PROJECT_ROOT).as_posix(),
        },
        {
            "report_element": "表2",
            "usage": "五组检测实验seed、epoch、imgsz、batch与最佳验证轮",
            "source_path": FINAL_TABLE.relative_to(PROJECT_ROOT).as_posix(),
        },
        {
            "report_element": "表3",
            "usage": "BASELINE-001内置指标与COCO复算指标",
            "source_path": "01_立项/阶段2_证据/stage2_test_evaluation.json;01_立项/阶段5_证据/stage5_experiment_metrics.csv",
        },
        {
            "report_element": "表4",
            "usage": "检测泛化结果、P2两seed和48轮预算对照",
            "source_path": FINAL_TABLE.relative_to(PROJECT_ROOT).as_posix(),
        },
        {
            "report_element": "表5",
            "usage": "SYS5S-001接口、完整测试复算与自动检查",
            "source_path": "01_立项/阶段5S_证据/forecast_result.json;01_立项/阶段5S_证据/阶段5S_资产检查.json;01_立项/阶段5S_证据/阶段5S审计.json",
        },
        {
            "report_element": "图2-图4",
            "usage": "训练损失、尺寸分桶与检测主结果",
            "source_path": "01_立项/阶段6_证据/主图冻结清单.csv;01_立项/阶段6_证据/最终主表.csv",
        },
        {
            "report_element": "图5-图6",
            "usage": "模块级联数据流与单窗口预测差",
            "source_path": "01_立项/阶段5S_证据/阶段5S预注册与接口设计.md;01_立项/阶段5S_证据/forecast_result.json;01_立项/阶段5S_证据/cascade_predictions.npz",
        },
    ]
    write_csv(CITATION_MAP, rows, ["report_element", "usage", "source_path"])


def main() -> int:
    if not TEMPLATE_DOCX.is_file():
        raise FileNotFoundError(TEMPLATE_DOCX)
    if sha256_file(TEMPLATE_DOCX) != TEMPLATE_SHA256:
        raise RuntimeError("报告模板副本哈希与冻结契约不一致")
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    BUILD_RECORD.parent.mkdir(parents=True, exist_ok=True)

    content_text = CONTENT_PATH.read_text(encoding="utf-8")
    percentages = validate_percentages(content_text)
    content = parse_content()
    prepare_report_figures()

    with tempfile.TemporaryDirectory(prefix="stage7_report_") as temp_dir:
        work_docx = Path(temp_dir) / "stage7-working.docx"
        doc = Document(TEMPLATE_DOCX)
        set_cover_title(doc)
        mark_template_table_headers(doc)
        anchors = prepare_headings(doc, content)
        stats = insert_sections(doc, content, anchors)
        doc.save(work_docx)
        package_patch = add_first_page_blank_header_and_update_fields(
            work_docx, OUTPUT_DOCX
        )

    build_citation_map()
    final_doc = Document(OUTPUT_DOCX)
    figure_hashes = {
        key: sha256_file(spec["path"]) for key, spec in FIGURE_SPECS.items()
    }
    figure_dimensions = {}
    for key, spec in FIGURE_SPECS.items():
        with Image.open(spec["path"]) as image:
            figure_dimensions[key] = list(image.size)
    payload = {
        "generated_at": now_iso(),
        "stage": "7",
        "report_title": REPORT_TITLE,
        "source_template": TEMPLATE_DOCX.relative_to(PROJECT_ROOT).as_posix(),
        "source_template_sha256": sha256_file(TEMPLATE_DOCX),
        "content_source": CONTENT_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "content_source_sha256": sha256_file(CONTENT_PATH),
        "output_docx": OUTPUT_DOCX.relative_to(PROJECT_ROOT).as_posix(),
        "output_docx_sha256": sha256_file(OUTPUT_DOCX),
        "output_size_bytes": OUTPUT_DOCX.stat().st_size,
        "percentages_in_content": percentages,
        "paragraphs_after_build": len(final_doc.paragraphs),
        "tables_after_build": len(final_doc.tables),
        "sections_after_build": len(final_doc.sections),
        "inline_shapes_after_build": len(final_doc.inline_shapes),
        "content_stats": stats,
        "figures": figure_hashes,
        "figure_dimensions": figure_dimensions,
        "package_patch": package_patch,
        "notes": [
            "报告由冻结模板副本创建，没有从空白DOCX重建。",
            "封面信息中的个人学号、姓名、班级和年级未由用户提供，保持模板空位。",
            "新增空白首页页眉部件，仅用于消除LibreOffice下封面异常顶部横线。",
            "其余页眉、页脚、样式、编号、主题、图片和关系以模板为准。",
        ],
    }
    write_json(BUILD_RECORD, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
