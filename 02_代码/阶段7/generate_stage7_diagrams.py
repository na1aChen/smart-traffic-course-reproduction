from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_ROOT = PROJECT_ROOT / "04_报告" / "图表"


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


def draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    color: str,
) -> None:
    from matplotlib.patches import FancyBboxPatch

    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.0,
        edgecolor="#7F8C8D",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=10,
        color="#17202A",
        wrap=True,
    )


def research_route() -> None:
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    top = [
        (0.03, "数据与来源锁版\nTT100K / YOLOv10"),
        (0.36, "基线复现\nBASELINE-001"),
        (0.69, "错误诊断\n尺寸 / 类别 / 工作点"),
    ]
    bottom = [
        (0.03, "P2单因子\n小目标高分辨率分支"),
        (0.36, "稳健性验证\n2 seed + 预算对照"),
        (0.69, "5S离线联调\n检测 -> 流量预测"),
    ]
    colors = ["#D6EAF8", "#D5F5E3", "#FCF3CF", "#FDEBD0", "#E8DAEF", "#D6EAF8"]
    for index, (x, text) in enumerate(top + bottom):
        draw_box(
            ax,
            x,
            0.63 if index < 3 else 0.12,
            0.28,
            0.25,
            text,
            colors[index],
        )
    arrow = {"arrowstyle": "-|>", "color": "#566573", "linewidth": 1.3}
    ax.annotate("", xy=(0.36, 0.755), xytext=(0.31, 0.755), arrowprops=arrow)
    ax.annotate("", xy=(0.69, 0.755), xytext=(0.64, 0.755), arrowprops=arrow)
    ax.annotate(
        "",
        xy=(0.17, 0.37),
        xytext=(0.83, 0.61),
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#566573",
            "linewidth": 1.0,
            "connectionstyle": "arc3,rad=-0.2",
        },
    )
    ax.annotate("", xy=(0.36, 0.245), xytext=(0.31, 0.245), arrowprops=arrow)
    ax.annotate("", xy=(0.69, 0.245), xytext=(0.64, 0.245), arrowprops=arrow)
    ax.text(
        0.5,
        0.96,
        "证据先行的六阶段研究路线",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.02,
        "阶段结论只在门槛与证据闭合后进入下一阶段",
        ha="center",
        va="center",
        fontsize=9,
        color="#566573",
    )
    fig.savefig(
        FIGURE_ROOT / "图1_研究路线.png",
        dpi=220,
        bbox_inches="tight",
        metadata={"Software": "stage7-report-v1.0.0"},
    )
    plt.close(fig)


def system_dataflow() -> None:
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    nodes = [
        (0.015, "检测事件\n6,780条 / 2,621图"),
        (0.215, "适配器\n96箱 / 标准化"),
        (0.415, "PEMS输入\n96 × 170"),
        (0.615, "TA-iTransformer\n严格加载旧权重"),
        (0.815, "预测输出\n96 × 170"),
    ]
    node_colors = ["#D6EAF8", "#EAECEE", "#D6EAF8", "#E8DAEF", "#D5F5E3"]
    for index, (x, text) in enumerate(nodes):
        draw_box(ax, x, 0.35, 0.17, 0.34, text, node_colors[index])
        if index < len(nodes) - 1:
            next_x = nodes[index + 1][0]
            ax.annotate(
                "",
                xy=(next_x - 0.006, 0.52),
                xytext=(x + 0.172, 0.52),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "#566573",
                    "linewidth": 1.4,
                },
            )
    ax.text(
        0.5,
        0.91,
        "检测结果到流量预测的离线模块级联",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.12,
        "图像ID顺序代理不等于真实时间；不重训、不修改权重，不声明端到端精度",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#C0392B",
    )
    fig.savefig(
        FIGURE_ROOT / "图5_模块级联数据流.png",
        dpi=220,
        bbox_inches="tight",
        metadata={"Software": "stage7-report-v1.0.0"},
    )
    plt.close(fig)


def main() -> int:
    configure_plotting()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    research_route()
    system_dataflow()
    print(f"diagrams generated: {FIGURE_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
