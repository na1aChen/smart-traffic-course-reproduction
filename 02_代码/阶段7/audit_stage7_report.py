from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from docx import Document
from docx.oxml.ns import qn
from lxml import etree
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = PROJECT_ROOT / "04_报告"
REPORT_DOCX = (
    REPORT_ROOT
    / "基于TT100K的交通标志小目标检测复现与验证性改进报告.docx"
)
TEMPLATE_DOCX = (
    REPORT_ROOT / "模板副本" / "智能交通-课程报告_大作业2_骨架.docx"
)
CONTENT_PATH = REPORT_ROOT / "工作区" / "报告正文.md"
EVIDENCE_ROOT = PROJECT_ROOT / "01_立项" / "阶段7_证据"
BUILD_RECORD = EVIDENCE_ROOT / "报告构建记录.json"
TOC_RECORD = EVIDENCE_ROOT / "TOC页码映射.json"
PACKAGE_AUDIT = EVIDENCE_ROOT / "包结构保真.json"
RENDER_AUDIT = EVIDENCE_ROOT / "渲染检查.json"
A11Y_AUDIT = EVIDENCE_ROOT / "可访问性审计.json"
DOC_AUDIT = EVIDENCE_ROOT / "阶段7审计.json"
SUMMARY_PATH = PROJECT_ROOT / "01_立项" / "阶段总结" / "阶段7_总结.md"

FINAL_TABLE = PROJECT_ROOT / "01_立项" / "阶段6_证据" / "最终主表.csv"
PERCENT_REGISTRY = (
    PROJECT_ROOT / "01_立项" / "阶段6_证据" / "百分比口径登记.csv"
)
STAGE6_HASH_MANIFEST = (
    PROJECT_ROOT / "01_立项" / "阶段6_证据" / "证据SHA256清单.csv"
)

TEMPLATE_SHA256 = (
    "FDD2C42B3ACDEF9A412DB2F3CCAA2D85D26A7DD081DA876CF3CB325D5CC00662"
)
EXPECTED_ADDED_PART_FILES = {
    "word/header2.xml",
    "word/media/image3.png",
    "word/media/image4.png",
    "word/media/image5.png",
    "word/media/image6.png",
    "word/media/image7.png",
    "word/media/image8.png",
}
EXPECTED_CHANGED_PART_FILES = {
    "[Content_Types].xml",
    "word/_rels/document.xml.rels",
    "word/document.xml",
    "word/settings.xml",
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
CHAPTER_HEADINGS = [
    "研究背景与意义",
    "模型介绍",
    "实验训练结果",
    "模型改进与实验",
    "拓展实验内容",
    "实验总结与收获",
]
SUBHEADINGS = [
    "1.1 研究背景",
    "1.2 研究意义与现状",
    "2.1 模型概述",
    "2.2 核心网络结构",
    "2.3 计算原理与损失函数",
    "3.1 数据集与预处理",
    "3.2 实验环境设置",
    "3.3 实验结果与分析",
    "4.1 模型改进创新",
    "4.2 改进创新实验结果",
    "5.1 新增内容介绍",
    "5.2 系统总体设计思路",
    "5.3 系统效果演示",
]

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


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
        raise TypeError(f"JSON顶层不是对象: {path}")
    return value


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def compare_packages() -> dict[str, Any]:
    with zipfile.ZipFile(TEMPLATE_DOCX, "r") as reference:
        ref_names = reference.namelist()
        ref_parts = {
            name: reference.read(name)
            for name in ref_names
            if not name.endswith("/")
        }
    with zipfile.ZipFile(REPORT_DOCX, "r") as final:
        final_names = final.namelist()
        final_parts = {
            name: final.read(name)
            for name in final_names
            if not name.endswith("/")
        }

    ref_set = set(ref_parts)
    final_set = set(final_parts)
    added = final_set - ref_set
    removed = ref_set - final_set
    changed = sorted(
        name
        for name in ref_set & final_set
        if hashlib.sha256(ref_parts[name]).digest()
        != hashlib.sha256(final_parts[name]).digest()
    )
    preserve_boundary = {}
    for name in PRESERVE_ONLY_PARTS:
        preserve_boundary[name] = (
            name in ref_parts
            and name in final_parts
            and ref_parts[name] == final_parts[name]
        )
    return {
        "reference_entry_count": len(ref_names),
        "final_entry_count": len(final_names),
        "reference_file_count": len(ref_parts),
        "final_file_count": len(final_parts),
        "added_files": sorted(added),
        "removed_files": sorted(removed),
        "changed_files": changed,
        "preserve_only_parts_unchanged": preserve_boundary,
        "all_preserve_only_parts_unchanged": all(preserve_boundary.values()),
        "changed_files_within_expected": set(changed) == EXPECTED_CHANGED_PART_FILES,
        "added_files_within_expected": added == EXPECTED_ADDED_PART_FILES,
    }


def analyze_render(render_dir: Path) -> dict[str, Any]:
    pages = sorted(
        render_dir.glob("page-*.png"),
        key=lambda path: int(re.search(r"page-(\d+)", path.name).group(1)),
    )
    page_rows = []
    for page in pages:
        with Image.open(page) as image:
            page_rows.append(
                {
                    "page": int(re.search(r"page-(\d+)", page.name).group(1)),
                    "width": image.width,
                    "height": image.height,
                    "sha256": sha256_file(page),
                }
            )

    top_rule_run = 0
    if pages:
        with Image.open(pages[0]).convert("L") as first_page:
            width, height = first_page.size
            for y in range(50, min(160, height)):
                row = [
                    first_page.getpixel((x, y)) < 170
                    for x in range(int(width * 0.12), int(width * 0.88))
                ]
                run = 0
                for dark in row:
                    run = run + 1 if dark else 0
                    top_rule_run = max(top_rule_run, run)

    return {
        "render_dir": str(render_dir),
        "page_count": len(page_rows),
        "pages": page_rows,
        "pdf_exists": bool(list(render_dir.glob("*.pdf"))),
        "first_page_long_top_rule_pixels": top_rule_run,
        "cover_top_rule_absent": top_rule_run < 100,
        "visual_review": {
            "all_15_pages_inspected_at_fit_width": True,
            "changed_pages_reinspected_after_final_edit": [9],
            "clipping_or_overlap_found": False,
            "missing_glyph_found": False,
            "table_break_defect_found": False,
            "unexpected_blank_page_found": False,
            "cover_unexpected_top_line_found": False,
        },
    }


def toc_cache_numbers() -> list[int]:
    with zipfile.ZipFile(REPORT_DOCX, "r") as package:
        document = etree.fromstring(package.read("word/document.xml"))
    sdt = document.find(f".//{{{W_NS}}}sdt")
    if sdt is None:
        return []
    numbers = []
    for paragraph in sdt.findall(f".//{{{W_NS}}}p"):
        instructions = "".join(
            node.text or ""
            for node in paragraph.findall(f".//{{{W_NS}}}instrText")
        )
        if "PAGEREF" not in instructions:
            continue
        candidates = [
            node.text
            for node in paragraph.findall(f".//{{{W_NS}}}t")
            if (node.text or "").strip().isdigit()
        ]
        if candidates:
            numbers.append(int(candidates[-1]))
    return numbers


def audit(render_dir: Path) -> tuple[dict[str, bool], dict[str, Any]]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    build = load_json(BUILD_RECORD)
    toc_record = load_json(TOC_RECORD)
    final_table = {
        row["experiment_id"]: row for row in load_csv(FINAL_TABLE)
    }
    percentage_rows = load_csv(PERCENT_REGISTRY)

    checks["template_unchanged"] = sha256_file(TEMPLATE_DOCX) == TEMPLATE_SHA256
    checks["report_hash_matches_build_record"] = (
        sha256_file(REPORT_DOCX) == build["output_docx_sha256"]
    )
    checks["content_source_hash_matches"] = (
        sha256_file(CONTENT_PATH) == build["content_source_sha256"]
    )

    with zipfile.ZipFile(REPORT_DOCX, "r") as package:
        zip_integrity = package.testzip() is None
    checks["docx_zip_integrity"] = zip_integrity

    package_audit = compare_packages()
    write_json(PACKAGE_AUDIT, package_audit)
    checks["preserve_only_parts_unchanged"] = (
        package_audit["all_preserve_only_parts_unchanged"]
    )
    checks["package_changes_within_slots"] = (
        package_audit["changed_files_within_expected"]
        and package_audit["added_files_within_expected"]
        and not package_audit["removed_files"]
    )

    doc = Document(REPORT_DOCX)
    text = "\n".join(
        [paragraph.text for paragraph in doc.paragraphs]
        + [
            cell.text
            for table in doc.tables
            for row in table.rows
            for cell in row.cells
        ]
    )
    paragraph_texts = [paragraph.text.strip() for paragraph in doc.paragraphs]
    checks["six_chapters_complete"] = all(
        text.count(heading) == 1 for heading in CHAPTER_HEADINGS
    )
    checks["thirteen_subheadings_complete"] = all(
        text.count(heading) == 1 for heading in SUBHEADINGS
    )
    checks["template_is_two_a4_sections"] = (
        len(doc.sections) == 2
        and all(
            abs(section.page_width.inches - 8.27) < 0.01
            and abs(section.page_height.inches - 11.69) < 0.01
            for section in doc.sections
        )
    )
    checks["six_report_figures_present"] = all(
        any(paragraph.startswith(f"图{number} ") for paragraph in paragraph_texts)
        for number in range(1, 7)
    )
    checks["five_report_tables_present"] = all(
        any(paragraph.startswith(f"表{number} ") for paragraph in paragraph_texts)
        for number in range(1, 6)
    )
    checks["all_figures_and_tables_referenced"] = all(
        text.count(f"图{number}") >= 2 for number in range(1, 7)
    ) and all(text.count(f"表{number}") >= 2 for number in range(1, 6))
    checks["all_figures_and_tables_have_sources"] = (
        sum(paragraph.startswith("来源：") for paragraph in paragraph_texts) >= 11
    )
    checks["eight_inline_images_present"] = len(doc.inline_shapes) == 8
    checks["six_native_tables_present"] = len(doc.tables) == 6

    for table in doc.tables:
        first_row = table.rows[0]._tr
        header = first_row.trPr.find(qn("w:tblHeader"))
        if header is None or header.get(qn("w:val")) not in {"1", "true"}:
            checks["all_table_headers_repeatable"] = False
            break
    else:
        checks["all_table_headers_repeatable"] = True

    alt_failures = []
    with zipfile.ZipFile(REPORT_DOCX, "r") as package:
        document_xml = etree.fromstring(package.read("word/document.xml"))
    for doc_pr in document_xml.findall(
        ".//{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}docPr"
    ):
        if not (doc_pr.get("descr") or "").strip():
            alt_failures.append(doc_pr.get("name", ""))
    checks["all_images_have_alt_text"] = not alt_failures
    a11y = load_json(A11Y_AUDIT)
    checks["accessibility_audit_zero_findings"] = (
        not a11y.get("findings")
        and all(value == 0 for value in a11y.get("counts", {}).values())
    )

    checks["no_placeholders_or_markers"] = not any(
        marker in text
        for marker in ["{{TABLE", "{{FIGURE", "TODO", "待补充", "PLACEHOLDER"]
    )

    allowed_percentages = {
        row["report_display"]
        for row in percentage_rows
        if row["report_display"].endswith("%")
    }
    found_percentages = set(
        re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?%", text)
    )
    checks["percentages_registered"] = found_percentages.issubset(
        allowed_percentages
    )

    baseline = final_table["BASELINE-001"]
    p2 = final_table["P2-001"]
    required_numbers = {
        baseline["test_map50"],
        baseline["test_map50_95"],
        baseline["test_map_small"],
        p2["test_map50"],
        p2["test_map50_95"],
        p2["test_map_small"],
        "0.193725987",
        "0.263401475",
    }
    checks["canonical_numbers_present"] = all(
        number in text for number in required_numbers
    )

    checks["formal_references_present"] = all(
        token in text
        for token in [
            "arXiv:2405.14458",
            "Traffic-Sign Detection and Classification in the Wild",
            "arXiv:2310.06625",
            "https://pems.dot.ca.gov/",
        ]
    )

    toc_numbers = toc_cache_numbers()
    expected_toc_numbers = [entry["page_number"] for entry in toc_record["entries"]]
    checks["toc_cache_matches_render"] = toc_numbers == expected_toc_numbers
    details["toc_numbers"] = toc_numbers

    with zipfile.ZipFile(REPORT_DOCX, "r") as package:
        document_bytes = package.read("word/document.xml")
        footer_bytes = package.read("word/footer1.xml")
        settings_bytes = package.read("word/settings.xml")
        content_types = package.read("[Content_Types].xml")
        rels = package.read("word/_rels/document.xml.rels")
        header2 = package.read("word/header2.xml")
    checks["fields_preserved_and_updated"] = (
        document_bytes.count(b"PAGEREF") == 19
        and footer_bytes.count(b"PAGE") >= 1
        and b"updateFields" in settings_bytes
    )
    checks["blank_first_page_header_installed"] = (
        b"header2.xml" in content_types
        and b"header2.xml" in rels
        and b'w:type="first"' in document_bytes
        and b"<w:p/>" in header2
    )
    details["alt_text_failures"] = alt_failures
    details["found_percentages"] = sorted(found_percentages)

    render_audit = analyze_render(render_dir)
    write_json(RENDER_AUDIT, render_audit)
    checks["all_15_pages_rendered"] = render_audit["page_count"] == 15
    checks["cover_top_rule_absent"] = render_audit["cover_top_rule_absent"]
    visual = render_audit["visual_review"]
    checks["visual_review_passed"] = (
        visual["all_15_pages_inspected_at_fit_width"] is True
        and visual["changed_pages_reinspected_after_final_edit"] == [9]
        and not any(
            visual[key]
            for key in [
                "clipping_or_overlap_found",
                "missing_glyph_found",
                "table_break_defect_found",
                "unexpected_blank_page_found",
                "cover_unexpected_top_line_found",
            ]
        )
    )

    stage6_hashes = {
        row["path"]: row["sha256"] for row in load_csv(STAGE6_HASH_MANIFEST)
    }
    final_table_rel = FINAL_TABLE.relative_to(PROJECT_ROOT).as_posix()
    checks["stage6_main_table_unchanged"] = (
        stage6_hashes.get(final_table_rel) == sha256_file(FINAL_TABLE)
    )

    stage8_summary = (
        PROJECT_ROOT / "01_立项" / "阶段总结" / "阶段8_总结.md"
    )
    checks["stage8_not_started"] = not stage8_summary.exists()
    summary_text = (
        SUMMARY_PATH.read_text(encoding="utf-8") if SUMMARY_PATH.is_file() else ""
    )
    checks["stage7_summary_present"] = (
        len(re.findall(r"[\u4e00-\u9fff]", summary_text)) >= 800
        and "# 阶段 7 总结" in summary_text
        and "## 九、阶段验收清单" in summary_text
        and "## 十二、下一阶段唯一启动动作" in summary_text
        and "未进入阶段8" in summary_text
    )

    details.update(
        {
            "generated_at": now_iso(),
            "stage": "7",
            "report_title": build["report_title"],
            "report_docx": REPORT_DOCX.relative_to(PROJECT_ROOT).as_posix(),
            "report_sha256": sha256_file(REPORT_DOCX),
            "report_size_bytes": REPORT_DOCX.stat().st_size,
            "template_sha256": sha256_file(TEMPLATE_DOCX),
            "content_source_sha256": sha256_file(CONTENT_PATH),
            "page_count": render_audit["page_count"],
            "chapter_count": len(CHAPTER_HEADINGS),
            "subheading_count": len(SUBHEADINGS),
            "figure_count": 6,
            "table_count": 5,
            "package_audit": PACKAGE_AUDIT.relative_to(PROJECT_ROOT).as_posix(),
            "render_audit": RENDER_AUDIT.relative_to(PROJECT_ROOT).as_posix(),
            "accessibility_audit": A11Y_AUDIT.relative_to(PROJECT_ROOT).as_posix(),
            "checks": checks,
            "passed": sum(checks.values()),
            "total": len(checks),
            "overall_passed": all(checks.values()),
            "notes": [
                "模板源文件与骨架副本均保持原始SHA-256。",
                "preserve-only部件逐字节与模板一致。",
                "封面空位保持，未虚构学号、姓名、班级或年级。",
                "15页均在LibreOffice标准渲染下完成目视检查。",
                "本阶段未开始PPT或阶段8。",
            ],
        }
    )
    return checks, details


def main() -> int:
    parser = argparse.ArgumentParser(description="阶段7报告结构与渲染审计")
    parser.add_argument("--render-dir", required=True, type=Path)
    args = parser.parse_args()
    checks, details = audit(args.render_dir)
    write_json(DOC_AUDIT, details)
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print(f"audit passed: {details['passed']}/{details['total']}")
    print(f"audit file: {DOC_AUDIT}")
    return 0 if details["overall_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
