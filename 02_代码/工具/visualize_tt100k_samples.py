from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deterministic visual QA grid from converted TT100K images."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def color_for(class_id: int) -> tuple[int, int, int]:
    palette = [
        (220, 50, 47),
        (38, 139, 210),
        (133, 153, 0),
        (181, 137, 0),
        (211, 54, 130),
        (108, 113, 196),
        (42, 161, 152),
        (203, 75, 22),
    ]
    return palette[class_id % len(palette)]


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    root = args.dataset_root.resolve()
    manifest = json.loads(
        (root / "manifests" / "split_manifest.json").read_text(encoding="utf-8")
    )
    rows = [row for row in manifest if row["split"] == args.split]
    rows.sort(
        key=lambda row: hashlib.sha256(
            f"{args.seed}:{row['image_id']}".encode()
        ).hexdigest()
    )
    rows = rows[: args.count]

    selection = json.loads((root / "class_selection.json").read_text(encoding="utf-8"))
    classes = selection["classes"]
    class_to_id = {
        str(key): int(value) for key, value in selection["class_to_id"].items()
    }
    id_to_class = {value: key for key, value in class_to_id.items()}

    tile_width, tile_height, label_height = 320, 320, 46
    columns = min(4, len(rows))
    grid_rows = (len(rows) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (tile_width * columns, (tile_height + label_height) * grid_rows),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for index, row in enumerate(rows):
        image_path = Path(row["output_image"])
        label_path = Path(row["label"])
        with Image.open(image_path) as source:
            tile = source.convert("RGB")
            tile = tile.resize((tile_width, tile_height), Image.Resampling.LANCZOS)

        tile_draw = ImageDraw.Draw(tile)
        category_ids = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            class_id_text, xc_text, yc_text, width_text, height_text = line.split()
            class_id = int(class_id_text)
            x_center, y_center, box_width, box_height = map(
                float, [xc_text, yc_text, width_text, height_text]
            )
            xmin = (x_center - box_width / 2) * tile_width
            xmax = (x_center + box_width / 2) * tile_width
            ymin = (y_center - box_height / 2) * tile_height
            ymax = (y_center + box_height / 2) * tile_height
            color = color_for(class_id)
            tile_draw.rectangle((xmin, ymin, xmax, ymax), outline=color, width=2)
            label = id_to_class[class_id]
            tile_draw.text(
                (max(0, xmin), max(0, ymin - 10)), label, fill=color, font=font
            )
            category_ids.append(class_id)

        column = index % columns
        row_index = index // columns
        x = column * tile_width
        y = row_index * (tile_height + label_height)
        canvas.paste(tile, (x, y))
        caption = (
            f"{row['image_id']} | {sum(1 for _ in category_ids)} boxes | "
            f"{len(set(category_ids))} classes"
        )
        draw.text((x + 4, y + tile_height + 5), caption, fill="black", font=font)
        draw.text(
            (x + 4, y + tile_height + 22),
            f"source={Path(row['source_image']).parent.name}",
            fill="black",
            font=font,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=92)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "split": args.split,
                "image_ids": [row["image_id"] for row in rows],
                "classes": classes,
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
