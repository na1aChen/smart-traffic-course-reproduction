"""Unit test for Stage 2's YOLO-txt -> COCO metric round trip."""

from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
from pathlib import Path

import yaml

SCRIPT = Path(__file__).resolve().parent / "stage2_runner.py"
spec = importlib.util.spec_from_file_location("stage2_runner", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load runner from {SCRIPT}")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def main() -> int:
    qa_root = runner.EXP_ROOT / "qa_coco_roundtrip"
    qa_root.mkdir(parents=True, exist_ok=True)
    output_dir = Path(tempfile.mkdtemp(prefix="perfect_gt_", dir=qa_root))
    prediction_dir = output_dir / "labels"
    prediction_dir.mkdir()

    data = yaml.safe_load(runner.DATA_YAML.read_text(encoding="utf-8"))
    data_root = Path(data["path"])
    label_dir = data_root / "labels" / "test"
    prediction_count = 0
    for source in sorted(label_dir.glob("*.txt")):
        rows = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(f"{line.strip()} 1.0")
                prediction_count += 1
        (prediction_dir / source.name).write_text(
            "\n".join(rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )

    metrics = runner.recompute_metrics(
        runner.DATA_YAML,
        "test",
        output_dir,
        prediction_dir,
    )
    checkpoint_path = qa_root / "中文路径保存测试" / "probe.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    runner.ultralytics_patches.torch_save(
        {"value": runner.torch.tensor([1.0])},
        checkpoint_path,
    )
    checkpoint_roundtrip = runner.torch.load(checkpoint_path)
    unicode_save_passed = bool(
        checkpoint_path.is_file()
        and runner.torch.equal(
            checkpoint_roundtrip["value"], runner.torch.tensor([1.0])
        )
    )
    result = {
        "description": "把测试集真实标注直接当作满分预测，验证 txt->COCO 往返。",
        "prediction_count": prediction_count,
        "unicode_checkpoint_save_passed": unicode_save_passed,
        "metrics": metrics,
        "passed": (
            metrics["images"] == 3071
            and metrics["categories"] == 45
            and metrics["map50"] > 0.999
            and metrics["map50_95"] > 0.999
            and unicode_save_passed
        ),
    }
    result_path = qa_root / "qa_coco_roundtrip.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (qa_root / "qa_coco_roundtrip.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
