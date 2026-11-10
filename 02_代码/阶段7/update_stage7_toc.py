from __future__ import annotations

import hashlib
import json
import re
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

from lxml import etree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DOCX = (
    PROJECT_ROOT
    / "04_报告"
    / "基于TT100K的交通标志小目标检测复现与验证性改进报告.docx"
)
BUILD_RECORD = PROJECT_ROOT / "01_立项" / "阶段7_证据" / "报告构建记录.json"
TOC_RECORD = PROJECT_ROOT / "01_立项" / "阶段7_证据" / "TOC页码映射.json"

TOC_PAGE_MAP = [
    ("一、研究背景与意义", 2),
    ("1.1 研究背景", 2),
    ("1.2 研究意义与现状", 3),
    ("二、模型介绍", 4),
    ("2.1 模型概述", 4),
    ("2.2 核心网络结构", 4),
    ("2.3 计算原理与损失函数", 5),
    ("三、实验训练结果", 5),
    ("3.1 数据集与预处理", 5),
    ("3.2 实验环境设置", 6),
    ("3.3 实验结果与分析", 7),
    ("四、模型改进与实验", 9),
    ("4.1 模型改进创新", 9),
    ("4.2 改进创新实验结果", 9),
    ("五、拓展实验内容", 11),
    ("5.1 新增内容介绍", 11),
    ("5.2 系统总体设计思路", 11),
    ("5.3 系统效果演示", 12),
    ("六、实验总结与收获", 14),
]

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def rewrite_docx(input_path: Path, output_path: Path) -> None:
    with zipfile.ZipFile(input_path, "r") as source:
        parts = {name: source.read(name) for name in source.namelist()}

    document = etree.fromstring(parts["word/document.xml"])
    sdt = document.find(f".//{{{W_NS}}}sdt")
    if sdt is None:
        raise RuntimeError("目录内容控件不存在")
    paragraphs = sdt.findall(f".//{{{W_NS}}}p")
    toc_entries = []
    for paragraph in paragraphs:
        instr = "".join(
            node.text or ""
            for node in paragraph.findall(f".//{{{W_NS}}}instrText")
        )
        match = re.search(r"PAGEREF\s+(_Toc\d+)", instr)
        if not match:
            continue
        toc_entries.append((paragraph, match.group(1)))
    if len(toc_entries) != len(TOC_PAGE_MAP):
        raise RuntimeError(
            f"目录PAGEREF数量不匹配: {len(toc_entries)} != {len(TOC_PAGE_MAP)}"
        )

    updated = []
    for (paragraph, anchor), (heading, page_number) in zip(
        toc_entries, TOC_PAGE_MAP, strict=True
    ):
        candidates = [
            node
            for node in paragraph.findall(f".//{{{W_NS}}}t")
            if (node.text or "").strip().isdigit()
        ]
        if not candidates:
            raise RuntimeError(f"目录项缺少缓存页码: {heading}")
        candidates[-1].text = str(page_number)
        updated.append(
            {
                "heading": heading,
                "anchor": anchor,
                "page_number": page_number,
            }
        )

    parts["word/document.xml"] = etree.tostring(
        document,
        xml_declaration=True,
        encoding="UTF-8",
        standalone=True,
    )
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as target:
        for name in sorted(parts):
            target.writestr(name, parts[name])

    TOC_RECORD.parent.mkdir(parents=True, exist_ok=True)
    with TOC_RECORD.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                {
                    "generated_at": datetime.now()
                    .astimezone()
                    .isoformat(timespec="seconds"),
                    "source_render": "report-render-v3",
                    "displayed_page_rule": "封面不计页码；目录显示第1页",
                    "entries": updated,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )

    build_record = json.loads(BUILD_RECORD.read_text(encoding="utf-8"))
    build_record["output_docx_sha256"] = sha256_file(output_path)
    build_record["output_size_bytes"] = output_path.stat().st_size
    build_record["toc_page_map"] = [
        {"heading": heading, "page_number": page} for heading, page in TOC_PAGE_MAP
    ]
    build_record["toc_cached_after_render"] = True
    with BUILD_RECORD.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(build_record, ensure_ascii=False, indent=2) + "\n"
        )


def main() -> int:
    expected_hash = json.loads(BUILD_RECORD.read_text(encoding="utf-8"))[
        "output_docx_sha256"
    ]
    if sha256_file(REPORT_DOCX) != expected_hash:
        raise RuntimeError("报告DOCX已变化，请先重新运行阶段7生成器")
    with tempfile.NamedTemporaryFile(
        prefix=".stage7-toc-",
        suffix=".docx",
        dir=REPORT_DOCX.parent,
        delete=False,
    ) as handle:
        temp_docx = Path(handle.name)
    try:
        rewrite_docx(REPORT_DOCX, temp_docx)
        temp_docx.replace(REPORT_DOCX)
    finally:
        if temp_docx.exists():
            temp_docx.unlink()
    print(f"toc updated: {REPORT_DOCX}")
    print(f"toc record: {TOC_RECORD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
