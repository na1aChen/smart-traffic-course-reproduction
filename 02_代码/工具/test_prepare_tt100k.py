from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "prepare_tt100k.py"
VALIDATOR = ROOT / "validate_tt100k_yolo.py"


def load_prepare_module():
    spec = importlib.util.spec_from_file_location("prepare_tt100k", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load prepare_tt100k.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_validator_module():
    spec = importlib.util.spec_from_file_location("validate_tt100k_yolo", VALIDATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load validate_tt100k_yolo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    prepare = load_prepare_module()
    validator = load_validator_module()
    with tempfile.TemporaryDirectory(prefix="tt100k_converter_") as temp_dir:
        root = Path(temp_dir) / "data"
        (root / "train").mkdir(parents=True)
        (root / "test").mkdir(parents=True)

        images = {}
        for image_id, split in [("1", "train"), ("2", "train"), ("3", "test")]:
            image_path = root / split / f"{image_id}.jpg"
            Image.new("RGB", (640, 480), (int(image_id), 40, 60)).save(image_path)
            objects = []
            for index, category in enumerate(prepare.TYPE45):
                xmin = 10 + index
                ymin = 20 + index
                objects.append(
                    {
                        "category": category,
                        "bbox": {
                            "xmin": xmin,
                            "ymin": ymin,
                            "xmax": xmin + 40,
                            "ymax": ymin + 30,
                        },
                    }
                )
            images[image_id] = {
                "id": int(image_id),
                "path": f"{split}/{image_id}.jpg",
                "objects": objects,
            }

        (root / "annotations.json").write_text(
            json.dumps({"imgs": images, "types": prepare.TYPE45}, ensure_ascii=False),
            encoding="utf-8",
        )
        (root / "train" / "ids.txt").write_text("1\n2\n", encoding="utf-8")
        (root / "test" / "ids.txt").write_text("3\n", encoding="utf-8")
        archive = Path(temp_dir) / "data.zip"
        archive.write_bytes(b"fixture archive")
        archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()

        output = Path(temp_dir) / "output"
        original_argv = __import__("sys").argv
        try:
            __import__("sys").argv = [
                str(SCRIPT),
                "--data-root",
                str(root),
                "--output-root",
                str(output),
                "--val-fraction",
                "0.5",
                "--seed",
                "42",
                "--archive",
                str(archive),
                "--archive-sha256",
                archive_sha256,
                "--force",
            ]
            prepare.main()
            prepare.main()
        finally:
            __import__("sys").argv = original_argv

        summary = json.loads(
            (output / "manifests" / "dataset_summary.json").read_text(encoding="utf-8")
        )
        assert summary["classes"] == 45
        assert summary["selected_boxes"] == {"train": 45, "val": 45, "test": 45}
        assert summary["archive_sha256"] == archive_sha256
        assert (output / "tt100k.yaml").exists()
        assert len(list((output / "images" / "train").glob("*.jpg"))) == 1
        assert len(list((output / "images" / "val").glob("*.jpg"))) == 1
        assert len(list((output / "images" / "test").glob("*.jpg"))) == 1
        dataset_yaml = yaml.safe_load(
            (output / "tt100k.yaml").read_text(encoding="utf-8")
        )
        assert dataset_yaml["names"] == prepare.TYPE45
        assert dataset_yaml["train"] == "images/train"
        assert dataset_yaml["val"] == "images/val"
        assert dataset_yaml["test"] == "images/test"

        try:
            __import__("sys").argv = [
                str(VALIDATOR),
                "--dataset-root",
                str(output),
            ]
            validator.main()
        finally:
            __import__("sys").argv = original_argv

        print("TT100K conversion smoke test passed")


if __name__ == "__main__":
    main()
