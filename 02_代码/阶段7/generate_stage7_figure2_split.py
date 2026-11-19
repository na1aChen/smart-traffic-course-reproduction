"""Generate three A4-friendly panels for BASELINE-001 loss curves."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
RESULTS_CSV = (
    PROJECT_ROOT / "03_实验" / "baseline_yolov10n" / "train" / "results.csv"
)
FIGURE_ROOT = PROJECT_ROOT / "04_报告" / "图表"

COMPONENTS = [
    {
        "key": "box",
        "title": "边界框损失",
        "columns": [
            "train/box_om",
            "val/box_om",
            "train/box_oo",
            "val/box_oo",
        ],
        "output": "图2a_边界框损失.png",
    },
    {
        "key": "cls",
        "title": "分类损失",
        "columns": [
            "train/cls_om",
            "val/cls_om",
            "train/cls_oo",
            "val/cls_oo",
        ],
        "output": "图2b_分类损失.png",
    },
    {
        "key": "dfl",
        "title": "DFL损失",
        "columns": [
            "train/dfl_om",
            "val/dfl_om",
            "train/dfl_oo",
            "val/dfl_oo",
        ],
        "output": "图2c_DFL损失.png",
    },
]

LINE_STYLES = ["-", "--", ":", "-."]
LINE_LABELS = ["训练-om", "验证-om", "训练-oo", "验证-oo"]
LINE_COLORS = ["#3A6EA5", "#F39C12", "#2E8B57", "#C44E52"]


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
            "font.size": 11,
        }
    )


def load_rows() -> list[dict[str, str]]:
    with RESULTS_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {str(key).strip(): str(value).strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def main() -> int:
    configure_plotting()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    epochs = [int(float(row["epoch"])) for row in rows]

    for component in COMPONENTS:
        fig, axis = plt.subplots(figsize=(8.2, 4.6))
        for column, linestyle, label, color in zip(
            component["columns"],
            LINE_STYLES,
            LINE_LABELS,
            LINE_COLORS,
            strict=True,
        ):
            axis.plot(
                epochs,
                [float(row[column]) for row in rows],
                linestyle=linestyle,
                linewidth=1.8,
                color=color,
                label=label,
            )

        axis.axvline(
            39,
            color="#6B7280",
            linewidth=1.2,
            linestyle=(0, (4, 3)),
            alpha=0.8,
            label="最佳验证轮=39",
        )
        axis.set_title(
            f"BASELINE-001的{component['title']}",
            fontsize=15,
            fontweight="bold",
            pad=12,
        )
        axis.set_xlabel("Epoch")
        axis.set_ylabel("损失值")
        axis.set_xlim(1, 40)
        axis.grid(axis="both", color="#D1D5DB", linewidth=0.7, alpha=0.45)
        axis.tick_params(direction="out", length=3)
        axis.legend(
            loc="upper right",
            frameon=False,
            fontsize=9,
            ncol=2,
        )
        fig.tight_layout()
        output = FIGURE_ROOT / component["output"]
        fig.savefig(
            output,
            dpi=240,
            bbox_inches="tight",
            metadata={"Software": "stage7-report-figure2-split-v1.0.0"},
        )
        plt.close(fig)
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
