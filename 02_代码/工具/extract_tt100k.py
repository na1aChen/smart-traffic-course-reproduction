from __future__ import annotations

import argparse
import os
import shutil
import time
import zipfile
from pathlib import Path, PurePosixPath


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely extract the official TT100K ZIP archive."
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def safe_target(output_root: Path, member_name: str) -> Path:
    normalized = member_name.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts:
        raise RuntimeError(f"Unsafe ZIP member path: {member_name}")
    target = (output_root / Path(*pure.parts)).resolve()
    output_abs = output_root.resolve()
    if output_abs != target and output_abs not in target.parents:
        raise RuntimeError(f"ZIP member escapes output root: {member_name}")
    return target


def main() -> None:
    args = parse_args()
    archive = args.archive.resolve()
    output_root = args.output_root.resolve()
    if not archive.exists():
        raise FileNotFoundError(archive)

    if output_root.exists() and any(output_root.iterdir()):
        if not args.force:
            raise RuntimeError(
                f"Output directory is not empty: {output_root}; pass --force to resume/extract"
            )
    else:
        output_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as zf:
        corrupt = zf.testzip()
        if corrupt is not None:
            raise RuntimeError(f"ZIP integrity failure at member: {corrupt}")

        members = [info for info in zf.infolist() if not info.is_dir()]
        total_bytes = sum(info.file_size for info in members)
        extracted_bytes = 0
        started = time.perf_counter()
        for index, info in enumerate(members, start=1):
            if info.filename.startswith("__MACOSX/") or info.filename.endswith(
                "/.DS_Store"
            ):
                continue
            target = safe_target(output_root, info.filename)
            if target.exists() and target.stat().st_size == info.file_size:
                extracted_bytes += info.file_size
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temp_target = target.with_name(target.name + ".part")
            with zf.open(info) as source, temp_target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            os.replace(temp_target, target)
            extracted_bytes += info.file_size
            if index == 1 or index % 250 == 0 or index == len(members):
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(
                    f"extracted={index}/{len(members)} "
                    f"bytes={extracted_bytes}/{total_bytes} "
                    f"rate_mib_s={extracted_bytes / 1024 / 1024 / elapsed:.2f} "
                    f"last={info.filename}",
                    flush=True,
                )

    print(f"extraction_complete={output_root}")


if __name__ == "__main__":
    main()
